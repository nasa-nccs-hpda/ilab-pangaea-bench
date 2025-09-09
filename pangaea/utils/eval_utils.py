import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import f1_score
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def config_cuda(
    cfg, backend="nccl", master_addr="localhost", master_port="12355"
):

    # Check if CUDA is available
    assert torch.cuda.is_available(), "CUDA is not available on this system."

    # Initialize distributed training if not already initialized
    if not dist.is_initialized():
        # Get distributed training parameters from environment vars
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        # Set required environment variables if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = master_addr
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = master_port

        # For single GPU training, we can skip distributed initialization
        if world_size == 1:
            print("Single GPU training detected, skipping distributed init.")
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
            return device, 0, 1

        # Initialize dist process group to allow distributed processes
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )

        print(
            f"Initialized distributed training: "
            f"rank={rank}, local_rank={local_rank}, "
            f"world_size={world_size}."
        )
    else:
        local_rank = dist.get_rank() % torch.cuda.device_count()
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    # Set the device for this process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Set memory management settings for better performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print(f"Using device: {device}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    return device, local_rank, world_size


def _get_state_dict(ckpt_dir, device):
    best_ckpts = list(ckpt_dir.rglob("*best.pth*"))
    assert len(best_ckpts) == 1, (
        f"More than 1 best model ckpt found in {ckpt_dir}."
        f"\nCheckpoint dir contents: {os.listdir(ckpt_dir)}"
    )
    # Load ckpt to model
    ckpt_path = best_ckpts[0]
    state_dict = torch.load(
        ckpt_path, map_location=device, weights_only=False
    )["model"]
    return state_dict


def _apply_checkpoint(state_dict, model):
    """Apply state dict checkpoint to model.

    The module attribute is used if DistributedDataParallel was used;
    otherwise, the whole model is used.
    """
    if hasattr(model, "module"):
        if "model" in state_dict:
            model.module.load_state_dict(state_dict["model"])
        else:
            model.module.load_state_dict(state_dict)
    else:
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
    return model


def load_apply_ckpt(ckpt_dir, device, model, logger, logger_verbosity):
    if logger_verbosity > 0:
        logger.info("Loading checkpoint...")
    state_dict = _get_state_dict(ckpt_dir, device)
    if logger_verbosity > 0:
        logger.info("State dict loaded from checkpoint.")
        logger.info("Applying checkpoint to model...")
    model = _apply_checkpoint(state_dict, model)
    model.eval()
    if logger_verbosity > 0:
        logger.info("Applied model checkpoint successfully.")
    return model


def _get_task(cfg):
    """From the config, extract which task we are performing."""
    task_dict = {
        "RegEvaluator": [
            "regression",
        ],
        "KNNClassificationEvaluator": ["knn_probe_multi_label", "knn_probe"],
        "LinearClassificationEvaluator": [
            "linear_classification_multi_label",
            "linear_classification",
        ],
        "SegEvaluator": ["segmentation"],
    }

    task_dict_query = cfg.task.evaluator._target_.split(".")[-1]
    task_list = task_dict.get(task_dict_query, [])

    if not task_list:
        return None

    if len(task_list) == 1:
        return task_list[0]

    # Check if this is a multi-label task
    is_multi_label = (
        hasattr(cfg.task.evaluator, "multi_label")
        and cfg.task.evaluator.multi_label
    )

    # Filter tasks based on multi-label flag
    matching_tasks = [
        task for task in task_list if ("multi_label" in task) == is_multi_label
    ]

    return matching_tasks[0] if matching_tasks else None


def _get_preds(cfg, outputs, task):
    pred_functions = {
        "regression": lambda x: x.squeeze(),
        "linear_classification": lambda x: torch.argmax(x, dim=1),
        "knn_probe": lambda x: torch.argmax(x, dim=1),
        "segmentation": (
            lambda x: (
                torch.sigmoid(x)
                if cfg.dataset.num_classes <= 1
                else torch.argmax(x, dim=1)
            )
        ),
    }

    # Handle multi_label special case
    if "multi_label" in task:
        return torch.sigmoid(outputs) > 0.5

    return pred_functions.get(task, lambda x: x)(outputs)


def _get_task_metric(task):
    """Get appropriate metric based on task."""
    metric_map = {
        "regression": "mse",
        "segmentation": "IoU",
        "linear_classification": "accuracy",
        "knn_probe": "accuracy",
    }

    # Handle multi_label special case
    if "multi_label" in task:
        return "f1"

    return metric_map.get(task, "")


def _get_metric(preds, targets, cfg, task, test_dict, device):
    """Append current batch's metric value to test_dict's metric field."""
    metric_name = test_dict["metric"]["name"]
    metric_value = test_dict["metric"]["value"]

    if metric_name == "mse":  # Regression
        metric_value += F.mse_loss(preds, targets)
    elif "IoU" in metric_name:  # Segmentation
        if cfg.dataset.num_classes > 1:  # Multiclass Segmentation
            iou = JaccardIndex(
                task="multiclass", num_classes=cfg.dataset.num_classes
            ).to(device)
        else:  # Binary Segmentation
            iou = JaccardIndex(task="binary").to(device)
        iou_score = iou(preds, targets)
        metric_value += iou_score
    elif metric_name == "accuracy":  # Cross-Entropy Loss
        metric_value += (preds == targets).float().mean()
    elif "multi_label" in task:  # F1-Score
        metric_value += f1_score(
            targets, (torch.sigmoid(preds) > 0.5), device=device
        )
    return metric_value


def test_loop(cfg, model, device, test_loader, logger):
    # Extract task from cfg so we can predict accurately
    task = _get_task(cfg)

    # Track values of evaluation
    test_dict = {
        "targets": [],
        "preds": [],
        "images": [],
        "metric": {"name": _get_task_metric(task), "value": 0.0},
    }

    # Load evaluation batches and predict, track metrics
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Test loop")):
            # Load all modalities of image (each is its own dict within image)
            images, targets = data["image"], data["target"]
            images = {k: v.to(device) for k, v in images.items()}
            targets = targets.to(device)

            # Model prediction processing depends on task
            outputs = model(images)
            preds = _get_preds(cfg, outputs, task).to(device)

            # Track images, targets, preds, metrics
            images = {k: v.cpu().numpy() for k, v in images.items()}
            test_dict["targets"].append(targets.cpu().numpy())
            test_dict["preds"].append(preds.cpu().numpy())
            test_dict["images"].append(images["optical"])
            test_dict["metric"]["value"] = _get_metric(
                preds, targets, cfg, task, test_dict, device
            )

    # After the loop, average the metric
    metric_name = test_dict["metric"]["name"]
    metric_value = test_dict["metric"]["value"]

    # Convert to tensor if it's not already
    if not isinstance(metric_value, torch.Tensor):
        metric_value = torch.tensor(metric_value, device=device)

    # Average the metric
    metric_value = metric_value / len(test_loader)

    # Update the test_dict with the averaged metric
    test_dict["metric"]["value"] = metric_value.item()

    # Log average metric
    logger.info(f"{metric_name} average over all eval samples: {metric_value}")

    for key in ["targets", "preds", "images"]:
        test_dict[key] = np.concatenate(test_dict[key], axis=0)

    return test_dict


def _get_cmap_from_task(cfg):
    task = _get_task(cfg)
    cmap = "viridis"
    is_discrete = False

    # Discrete cmaps for class/seg
    if task == "classification" or task == "segmentation":
        is_discrete = True
        if (cfg.dataset.num_classes > 10) and (cfg.dataset.num_classes < 20):
            cmap = "tab20b"
        elif cfg.dataset.num_classes <= 10:
            cmap = "tab10"
        else:
            warnings.warn(
                ">20 classes detected for seg/class task. "
                "Using viridis colormap."
            )
    elif task == "change_detection":
        cmap = "RdBu_r"
    elif task in ["knn_probe", "knn_probe_multi_label"]:
        cmap = "YlGnBu"  # Yellow-Green-Blue, good for distances/probabilities

    return cmap, is_discrete


def _prepare_tensor_data(targets, preds, images):
    """Prepare tensor data by removing extra dimensions."""
    # Make targets and preds 3D tensors if they are not
    if targets.ndim > 3:  # B, H, W
        targets = np.squeeze(targets, axis=0)
    if preds.ndim > 3:  # B, H, W
        preds = np.squeeze(preds, axis=0)
    if images.ndim > 4:  # B, C, H, W
        images = np.squeeze(images, axis=2)

    return targets, preds, images


def _get_discrete_cmap(n_classes, cmap_name):
    base_cmap = plt.cm.get_cmap(cmap_name)
    # Always use the same color indices for the same class values
    # This ensures class 0 always gets color 0, class 1 gets color 1, etc.
    colors = []
    for i, class_val in enumerate(range(n_classes)):
        # Map colors to indices based on n_classes
        color_idx = int(class_val) % base_cmap.N
        colors.append(base_cmap(color_idx))

    return ListedColormap(colors)


def _setup_discrete_colormap_and_norm(cfg, targets, preds, base_cmap):
    """Setup colormap and normalization for discrete classification data."""
    n_classes = cfg.dataset.num_classes

    # Use the discrete colormap function
    cmap = _get_discrete_cmap(n_classes, base_cmap)

    # Get unique classes for mapping
    unique_classes = np.sort(np.unique(np.concatenate([targets, preds])))

    # Map class values to sequential positions for BoundaryNorm
    class_to_position = {cls: i for i, cls in enumerate(unique_classes)}
    targets_viz = np.vectorize(class_to_position.get)(targets)
    preds_viz = np.vectorize(class_to_position.get)(preds)

    # Create discrete normalization with boundaries between classes
    boundaries = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(boundaries, ncolors=n_classes)

    return cmap, norm, targets_viz, preds_viz, unique_classes


def _setup_continuous_colormap_and_norm(targets, preds, base_cmap):
    """Setup colormap and normalization for continuous data."""
    # For continuous data, use original normalization
    all_data = np.concatenate([targets, preds])
    vmin, vmax = all_data.min(), all_data.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    return base_cmap, norm, targets, preds


def _create_heatmap_subplot(ax, data, cmap, norm, title, unique_classes=None):
    """Create a single heatmap subplot with colorbar."""
    im = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis("off")

    fig = ax.get_figure()
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
    )

    # For discrete data, set colorbar ticks to show class labels
    if unique_classes is not None:
        # Set ticks at the center of each color segment
        tick_positions = np.arange(len(unique_classes))
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(unique_classes)


def plot_results_heatmap(cfg, targets, preds, images, save_dir, png_prefix):
    """Plot heatmap comparison of targets vs predictions."""
    # Prepare tensor data
    targets, preds, images = _prepare_tensor_data(targets, preds, images)

    # Plot 5 samples by default
    num_samples = 5

    # Get task from config to inform color choices
    base_cmap, is_discrete = _get_cmap_from_task(cfg)

    # Setup colormap and normalization based on data type
    if is_discrete:
        cmap, norm, targets_viz, preds_viz, unique_classes = (
            _setup_discrete_colormap_and_norm(cfg, targets, preds, base_cmap)
        )
    else:
        cmap, norm, targets_viz, preds_viz = (
            _setup_continuous_colormap_and_norm(targets, preds, base_cmap)
        )
        unique_classes = None

    # Prepare batch data
    batch_targets = targets_viz[:num_samples]
    batch_preds = preds_viz[:num_samples]
    batch_size = batch_targets.shape[0]
    nrows, ncols = (2, batch_size)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 6))

    if ncols == 1:  # Handle case of single pair (ncols=1)
        axes = axes.reshape(2, 1)

    # Create heatmaps
    for j in range(batch_size):
        # Top row: Targets
        _create_heatmap_subplot(
            axes[0, j], batch_targets[j], cmap, norm, "Target", unique_classes
        )

        # Bottom row: Predictions
        _create_heatmap_subplot(
            axes[1, j],
            batch_preds[j],
            cmap,
            norm,
            "Prediction",
            unique_classes,
        )

    # Finalize and save
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{png_prefix}.png")
    plt.savefig(save_path)
    return fig


def _data_to_df(targets, predictions):
    """Create Pandas DataFrame with flattened predictions."""
    # Convert to numpy arrays directly without list appending and concatenation
    all_targets = targets.flatten()
    all_preds = predictions.flatten()

    # Calculate R² directly from the arrays
    r2 = np.corrcoef(all_targets, all_preds)[0, 1] ** 2
    print(f"R² calculated directly: {r2:.4f}")

    # Create the DataFrame more directly
    df = pd.DataFrame({"actual": all_targets, "predicted": all_preds})

    return df


def _scatter_plot(
    val_df, bins=50, title="Target vs Predicted Data Similarity", cmap="plasma"
):
    """
    Create a single bin2d plot using matplotlib with plasma colormap
    and properly positioned colorbar.
    """
    # Determine range for the plot
    min_val = min(val_df["actual"].min(), val_df["predicted"].min())
    max_val = max(val_df["actual"].max(), val_df["predicted"].max())

    # Calculate metrics
    val_r2 = np.corrcoef(val_df["actual"], val_df["predicted"])[0, 1] ** 2
    val_rmse = np.sqrt(np.mean((val_df["actual"] - val_df["predicted"]) ** 2))

    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Create the histogram2d plot
    h = ax.hist2d(
        val_df["actual"],
        val_df["predicted"],
        bins=bins,
        cmap=cmap,
        norm=plt.matplotlib.colors.LogNorm(),
    )

    # Add the 1:1 line
    ax.plot([min_val, max_val], [min_val, max_val], "b--", linewidth=1.5)

    # Set labels and title
    ax.set_xlabel("Actual Data", fontsize=12)
    ax.set_ylabel("Predicted Data", fontsize=12)
    ax.set_title(
        f"Validation\nR² = {val_r2:.3f}, RMSE = {val_rmse:.3f}", fontsize=14
    )

    # Set axis properties
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label("Count (log scale)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Add main title
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Adjust layout
    plt.tight_layout()

    return fig


def _plot_results_scatter(cfg, targets, predictions, save_dir, png_prefix):
    """Evaluate model and create comparison visualization."""
    print("Collecting validation predictions...")
    val_df = _data_to_df(targets, predictions)
    print("Creating visualization...")
    # Use matplotlib with plasma colormap
    fig = _scatter_plot(val_df)
    print(f"Validation samples: {len(val_df)}")
    # Calculate and print additional metrics
    val_r2 = np.corrcoef(val_df["actual"], val_df["predicted"])[0, 1] ** 2
    val_rmse = np.sqrt(np.mean((val_df["actual"] - val_df["predicted"]) ** 2))
    print(f"Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
    # Save the figure
    save_path = os.path.join(save_dir, f"{png_prefix}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, val_df


def _plot_confusion_matrix_from_df(
    df,
    class_names=None,
    figsize=(10, 8),
    normalize=False,
    title="Confusion Matrix",
):
    """
    Plot confusion matrix using seaborn from a pandas DataFrame
    """
    # Calculate confusion matrix from DataFrame
    cm = confusion_matrix(df["actual"], df["predicted"])

    # Determine class names if not provided
    if class_names is None:
        all_classes = sorted(
            set(
                np.concatenate(
                    [df["actual"].unique(), df["predicted"].unique()]
                )
            )
        )
        class_names = [str(c) for c in all_classes]

    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    # Create figure with optimized settings
    fig, ax = plt.subplots(figsize=figsize)  # Use subplots instead of figure()

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,  # Specify the axis
    )

    # Set labels and title on the axis object
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    return fig, cm


def _plot_results_conf_matrix(cfg, targets, predictions, save_dir, png_prefix):
    """Plot confusion matrix and other metrics for segmentation results"""

    # Convert to DataFrame
    df = _data_to_df(targets, predictions)

    # Sample large datasets for faster plotting
    if len(df) > 100000:
        print(
            f"Dataset is large ({len(df)} samples), sampling 50k for visualization..."
        )
        df = df.sample(n=50000, random_state=42)

    # Get class list from config
    class_names = [f"Class {i+1}" for i in range(cfg.dataset.num_classes)]

    print("Generating normalized confusion matrix...")

    # Turn off interactive mode for performance
    was_interactive = plt.isinteractive()
    plt.ioff()

    try:
        # Create the confusion matrix plot
        fig, cm_norm = _plot_confusion_matrix_from_df(
            df,
            class_names=class_names,
            normalize=True,
            title="Normalized Segmentation Confusion Matrix",
            figsize=(
                min(12, max(8, len(class_names) * 0.8)),
                min(12, max(8, len(class_names) * 0.8)),
            ),  # Dynamic sizing
        )

        # Get the current axis
        ax = fig.gca()

        # Rotate labels if many classes
        if len(class_names) > 8:
            ax.tick_params(axis="x", rotation=90, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
        elif len(class_names) > 5:
            ax.tick_params(axis="x", rotation=45, labelsize=9)

        # Add note at bottom
        fig.text(
            0.5,
            0.02,
            "* See classification report below",
            ha="center",
            fontsize=9,
            style="italic",
        )

        # Adjust layout once at the end
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        # Save the figure
        save_path = os.path.join(save_dir, f"{png_prefix}.png")
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

        print(f"Confusion matrix saved to: {save_path}")

        # Show the plot in Jupyter
        plt.show()

    finally:
        # Always restore interactive mode and clean up
        if was_interactive:
            plt.ion()

        # Close the figure to free memory
        plt.close(fig)

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            df["actual"], df["predicted"], target_names=class_names
        )
    )

    return fig, df


def plot_results_variable(cfg, targets, predictions, save_dir, png_prefix):
    task = _get_task(cfg)

    # Scatter plots for class./regress. tasks
    if "classification" in task or "regression" in task:
        return _plot_results_scatter(
            cfg, targets, predictions, save_dir, png_prefix
        )
    # Conf. matrix for seg. tasks
    elif task == "segmentation":
        return _plot_results_conf_matrix(
            cfg, targets, predictions, save_dir, png_prefix
        )
    elif "knn" in task:
        pass
    elif "change_detection" in task:
        pass
