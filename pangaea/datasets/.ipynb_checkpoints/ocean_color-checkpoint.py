import torch
import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from pathlib import Path
from osgeo import gdal
import torchvision.transforms as T
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.engine.data_preprocessor import PBMinMaxNorm

class OceanColorDataset(RawGeoFMDataset):
    """
    Dataset of MOD021KM Aqua Data. For now this uses .npy chip files.
    """

    def __init__(
        self,
        num_inputs: int,
        num_targets: int,
        # inherited from RawGeoFMDataset
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
        super(OceanColorDataset, self).__init__(
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

        self.samples = self.gather_files(root_path)
        self.img_size = img_size
        self.transform = PBMinMaxNorm()
        self.num_inputs = num_inputs
        self.num_targets = num_targets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next item in the dataset. Load sample from file,
        apply transforms to the entire sample, then extract inputs and targets.
        """
        # load and resize sample image
        image = self.resize_img(self.samples[index])

        # apply transform
        transformed = self.transform(image)

        # extract inputs and target(s)
        inputs = transformed[:self.num_inputs]  # Add T dim
        target = transformed[
            self.num_inputs:self.num_inputs + self.num_targets]
        target = target.squeeze()

        outputs = {
            "image" : {
                "optical": inputs,
            }, 
            "target": target,
            "metadata": {},
        }

        return outputs
    
    def resize_img(self, image: np.ndarray) -> np.ndarray:
        """Prithvi requires a specific image size, so we reshape before transforming."""
        image = np.expand_dims(image, axis=0)  # Add batch dim for resize
        resized = F.interpolate(
            torch.from_numpy(image), 
            size=(self.img_size, self.img_size), 
            mode='bilinear', 
            align_corners=False
        )
        # shape needs to to be (C, T, H, W)
        resized = resized.permute(1, 0, 2, 3)
        return resized.numpy()

    @staticmethod
    def download(self, silent=False):
        pass
    
    def gather_files(self, data_path: str) -> list[str]:
        """
        Finds all filenames in data_path and all its subdirs. 
        Loads them into a numpy array of samples.
        Only looks 1 subdirectory deep (e.g. doesn't look recursively
        in directories of directories). 
    
        Args:
            self: self
            data_path: string filepath where data is stored
        Returns:
            numpy.array of loaded samples from data_path and all of its subdirs
        """
        full_path = os.path.abspath(data_path)
        filenames = self.examine_dir(data_path)
        for subdir_name in self.find_subdirs(data_path):
            filenames = filenames + self.examine_dir(subdir_name)

        samples = [np.load(fn) for fn in filenames if fn.endswith('.npy')]
        return np.array(samples)

    def examine_dir(self, path: str) -> list[str]:
        """Finds all filenames in a given path."""
        filenames = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                filenames.append(item_path)
        return filenames

    def find_subdirs(self, path: str) -> list[str]:
        """Finds all directories in a given path."""
        return [
            os.path.join(path, item)
            for item in os.listdir(path)
            if os.path.isdir(os.path.join(path, item))
        ]
