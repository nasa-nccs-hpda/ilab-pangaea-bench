import json
import os
import glob

import numpy as np
import torch

from pangaea.datasets.base import RawGeoFMDataset

class Landsat_NLCD(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
    ):
        """Initialize the dataset for numpy files."""
        super(Landsat_NLCD, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )
    
        assert split in ["train", "val", "test"], "Split must be train, val or test"
        
        # Since you're only using train directory, ignore the split parameter
        # and always use the train directory
        self.images_folder = os.path.join(root_path, "images")
        self.labels_folder = os.path.join(root_path, "labels")
        
        # Get all image files
        image_files = glob.glob(os.path.join(self.images_folder, "image_*.npy"))
        image_files.sort()
        
        # Extract file IDs for pairing with labels
        self.file_ids = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            file_id = filename.replace("image_", "").replace(".npy", "")
            
            # Check if corresponding label exists
            label_path = os.path.join(self.labels_folder, f"label_{file_id}.npy")
            if os.path.exists(label_path):
                self.file_ids.append(file_id)
        
        print(f"Found {len(self.file_ids)} paired image-label files in {self.images_folder}")

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get the item at index i.
    
        Args:
            i (int): index of the item.
    
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {"optical": torch.Tensor},
            "target": torch.Tensor,
             "metadata": dict}.
        """
    
        file_id = self.file_ids[i]  
        
        # Load image (224, 224, 7)
        image_path = os.path.join(self.images_folder, f"image_{file_id}.npy")
        image_array = np.load(image_path)  # Shape: (224, 224, 7)
        
        # Load label (244, 244)
        label_path = os.path.join(self.labels_folder, f"label_{file_id}.npy")
        label_array = np.load(label_path)  # Shape: (224, 224)
        
        # # DEBUG: Check what's in the labels
        # if i == 0:  # Only print for first item
        #     print(f"Label dtype: {label_array.dtype}")
        #     print(f"Label shape: {label_array.shape}")
        #     print(f"Label unique values: {np.unique(label_array)}")
        #     print(f"Label min/max: {np.nanmin(label_array)}/{np.nanmax(label_array)}")
        #     print(f"Has NaN: {np.isnan(label_array).any()}")
        #     print(f"Has inf: {np.isinf(label_array).any()}")
        #     print(f"Sample values: {label_array.flatten()[:10]}")
        
        # Convert to tensors and rearrange dimensions
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        # Convert labels to integers (fix for the bincount error)
        label_tensor = torch.from_numpy(label_array.astype(np.int64))
        
        image_tensor = image_tensor.unsqueeze(1)
        
        return {
            "image": {
                "optical": image_tensor,
            },
            "target": label_tensor,
            "metadata": {"file_id": file_id},
        }

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.file_ids)

    @staticmethod
    def download():
        """No download needed for local numpy files."""
        print("Data is already local - no download needed")
        pass