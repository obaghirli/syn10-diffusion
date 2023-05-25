import numpy as np
from pathlib import Path
from typing import Union, Optional
from abc import ABC, abstractmethod

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from syn10_diffusion.utils import seed_all

seed_all()


class ABCDataset(Dataset, ABC):
    def __init__(
            self,
            image_dir: Union[str, Path],
            annotation_dir: Optional[Union[str, Path]] = None
    ):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = None
        self.annotation_files = None
        self.shard_id = dist.get_rank()
        self.num_shards = dist.get_world_size()

        assert self.image_dir is not None
        if isinstance(self.image_dir, str):
            self.image_dir = Path(self.image_dir)
        if not self.image_dir.exists():
            raise RuntimeError(f"Dataset image directory {self.image_dir} not found")
        self.load_image_files()

        if self.annotation_dir is not None:
            if isinstance(self.annotation_dir, str):
                self.annotation_dir = Path(self.annotation_dir)
            if not self.annotation_dir.exists():
                raise RuntimeError(f"Dataset annotation directory {self.annotation_dir} not found")
            self.load_annotation_files()

    def load_image_files(self):
        self.image_files = sorted(list(self.image_dir.glob('**/*.npy')))[self.shard_id:][::self.num_shards]
        assert len(self.image_files) >= self.num_shards

    def load_annotation_files(self):
        self.annotation_files = sorted(list(self.annotation_dir.glob('**/*.npy')))[self.shard_id:][::self.num_shards]
        assert len(self.annotation_files) >= self.num_shards
        assert len(self.image_files) == len(self.annotation_files)

    @abstractmethod
    def process_image(self, image):
        raise NotImplementedError

    @abstractmethod
    def process_annotation(self, annotation):
        raise NotImplementedError

    def __getitem__(self, index):
        image = self.process_image(np.load(str(self.image_files[index])))
        if self.annotation_dir is not None:
            image_filename = self.image_files[index]
            annotation_filename = self.annotation_files[index]
            assert image_filename.stem == annotation_filename.stem
            annotation = self.process_annotation(np.load(str(self.annotation_files[index])))
            return image, annotation
        return image

    def __len__(self):
        return len(self.image_files)


class FIDTransform:
    def __init__(self, image_max_value=255, image_min_value=0):
        self.image_max_value = image_max_value
        self.image_min_value = image_min_value

    def __call__(self, image):
        image = torch.from_numpy(image)
        image = TF.resize(image, size=[299, 299], interpolation=TF.InterpolationMode.NEAREST, antialias=True)
        image = (image - self.image_min_value) / (self.image_max_value - self.image_min_value)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image


class FIDDataset(ABCDataset):
    def __init__(self, image_dir, image_max_value=255, image_min_value=0):
        self.image_dir = image_dir
        self.image_max_value = image_max_value
        self.image_min_value = image_min_value
        super().__init__(image_dir)

    def process_image(self, image: np.ndarray):
        assert isinstance(image, np.ndarray), f"Image must be numpy array, got {type(image)}"
        assert image.ndim == 3, f"Image must be 3D, got {image.ndim}D"
        assert image.shape[0] == 3, f"Image must have 3 channels (channel first), got {image.shape[0]}"
        assert np.issubdtype(image.dtype, np.integer), f"Image must be integer type, got {image.dtype}"
        assert np.all(image >= self.image_min_value) and np.all(image <= self.image_max_value)

        image = image.astype(np.float32)
        preprocess = transforms.Compose([
            FIDTransform(
                image_max_value=self.image_max_value,
                image_min_value=self.image_min_value
            )
        ])
        image = preprocess(image)
        return image

    def process_annotation(self, annotation):
        return


def load_fid_dataset(
        image_dir: Union[str, Path],
        batch_size: int,
        image_max_value: int,
        image_min_value: int,
        shuffle: bool,
        drop_last: bool
):
    dataset = FIDDataset(image_dir, image_max_value=image_max_value, image_min_value=image_min_value)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=1)
    return loader


