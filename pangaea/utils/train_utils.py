from omegaconf import OmegaConf, open_dict
import os

defaults = {
    "dataset": {"auto_download": True, "ignore_index": -1},
    "encoder":    {"input_size": "${dataset.img_size}",
                   "input_bands": "${dataset.bands}"},
    "decoder": {"finetune": True, "num_classes": 1},
}


def get_folder_options(base_path):
    path = os.path.join(("ilab-pangaea-bench/configs"), base_path)
    return [
        os.path.splitext(fn)[0]
        for fn in os.listdir(path)
        if ".yaml" in fn
    ]


def is_valid_override(value):
    """Check if a config override value is valid."""
    # Always keep boolean False
    if value is False:
        return True

    # Reject empty strings
    if value == "":
        return False

    # Reject negative numbers
    if isinstance(value, (int, float)) and value < 0:
        return False

    # Keep everything else that's truthy
    return bool(value)


def build_config(
    config_type: str, params: dict, cfg: dict = {}, override: bool = False
):
    """
    Build a YAML configuration from user-defined parameters and stock dataset.

    This function creates or updates a configuration dictionary based on the
    provided parameters and an optional existing configuration.

    Args:
        config_type (str): Type of configuration, corresponding to the
            Pangaea config directory name.
        params (dict): User-defined parameters from the notebook.
        cfg (dict, optional): Existing configuration to build upon.
            Defaults to an empty dictionary.

    Returns:
        cfg_updated: The built or updated config dict.
        yaml_filename = Filename cfg was saved to, without .yaml extension
    """
    # Add default values to user-defined params
    print(f"original cfg: {cfg}")
    print(f"user params: {params}")
    params.update(defaults.get(config_type, {}))

    # Convert to DictConfig if it's a regular dict
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    # Update cfg with user params
    with open_dict(cfg):
        if override:  # Don't merge configs if we don't wish to override
            cfg_updated = OmegaConf.merge(cfg, params)
        else:
            cfg_updated = cfg
    print(f"merged cfg: {cfg_updated}")

    # Automatically generate filename from _target_ field
    target = (
        params.get("_target_") if override else
        cfg.get("_target_") or
        "custom_preprocessing"
    )

    # Save cfg to repo "configs" dir
    base_filename = target.split(".")[-1].lower()
    yaml_filename = f"{base_filename}.yaml"
    config_dir = f"ilab-pangaea-bench/configs/{config_type}"
    full_path = os.path.join(config_dir, yaml_filename)

    # Save config and return filename
    with open(full_path, "w", encoding="utf-8") as file:
        OmegaConf.save(cfg_updated, file)
    print(f"Config saved to {yaml_filename}")

    return cfg_updated, yaml_filename.split(".")[0]


def print_cfg_rec(cfg, indent=0):
    cfg = dict(cfg)

    def _build_string(cfg, indent=0):
        result = []
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                result.append('    ' * indent + str(key) + ':')
                if isinstance(value, (dict, list)):
                    result.append(_build_string(value, indent + 1))
                else:
                    result.append('    ' * (indent + 1) + str(value))
        elif isinstance(cfg, list):
            for i, item in enumerate(cfg):
                result.append('    ' * indent + f"[{i}]:")
                if isinstance(item, (dict, list)):
                    result.append(_build_string(item, indent + 1))
                else:
                    result.append('    ' * (indent + 1) + str(item))
        return '\n'.join(result)

    print(_build_string(cfg, indent))