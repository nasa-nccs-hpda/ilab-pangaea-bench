import os
import sys
import torch
import numpy as np
import geopandas as gpd
import rioxarray as rxr
from pathlib import Path
import torch.nn.functional as F
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
        gpkg_filename: str,
    ):
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

        # dataset parameters for filenames
        self.gpkg_filename = gpkg_filename
        self.dataset_gdf = gpd.read_file(
            self.gpkg_filename)

        # images and labels list
        self.mask_list = self.dataset_gdf.filename.tolist()
        self.image_list = self.dataset_gdf.filename.str.replace(
            "/labels/", "/images/", regex=False).tolist()

    def __len__(self):
        # Return the total number of samples
        return len(self.image_list)

    def __getitem__(self, index):
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {
                "optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                },
            "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
            regression datasets.,
            "metadata": dict}.
        """
        # Load your data and labels here
        # image = self._load_file(self.image_list[index])  # Load image
        # target = self._load_file(self.mask_list[index])  # Load target label or mask
        image, target = self._load(self[index])


        # Convert to tensors ## They already are
        # image = torch.tensor(image, dtype=torch.int32)#.unsqueeze(1)
        # target = torch.tensor(target, dtype=torch.int16).squeeze()
        
        # Only do RGB bands
        red_channel = image[:, :, 3]    # 4th channel
        green_channel = image[:, :, 2]  # 3rd channel  
        blue_channel = image[:, :, 1]   # 2nd channel
            
        # Stack into RGB array
        image = np.stack([red_channel, green_channel, blue_channel], axis=-1
                )

        # Add batch dimension for interpolation (NCHW)
        image = image.unsqueeze(0)  # shape: (1, 3, 64, 64)

        # Resize image to 224×224 (expected by PRITHVI) ## Already is 244x244
        # image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        #image = image.squeeze(0)  # back to (3, 224, 224)

        # Resize target if needed (optional, depends on task)
        # target = F.interpolate(target.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()

        #print(image.shape, target.shape)
        #image = image.unsqueeze(1)

        # Expand temporal dimension: (C, H, W) → (C, 1, H, W)
        # image = image.unsqueeze(0)

        # Resize spatial: (C, 1, H, W) → (C, 1, 224, 224)
        #image = F.interpolate(image, size=(1, 224, 224), mode='bilinear', align_corners=False)

        # Reorder to (C, T, H, W) → (T, C, H, W)
        image = image.permute(1, 0, 2, 3)

        #print(image.shape, target.shape)

        return {
            'image': {'optical': image},
            'target': target,
            'metadata': {}
        }

    @staticmethod
    def download(self, silent=False):
        # Implement if your dataset requires downloading
        pass

    def _load_file(self, path: Path):
        if Path(path).suffix == '.npy':
            data = np.load(path)
        elif Path(path).suffix == '.tif':
            data = rxr.open_rasterio(path).to_numpy()
        else:
            sys.exit('Non-recognized dataset format. Expects npy or tif.')
        return data

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list


if __name__ == '__main__':

    # set test data paths
    data_path = '/explore/nobackup/people/mfrost2/lscratch/EO_FM/data/Landsat_NLCD'
    train_filename = os.path.join(
        data_path, 'train_dataset.npz')
    test_filename = os.path.join(
        data_path, 'test_dataset.npz')
    val_filename = os.path.join(
        data_path, 'val_dataset.npz')

    # set dataset object
    train_dataset = train_filename
    # iterate for a small test
