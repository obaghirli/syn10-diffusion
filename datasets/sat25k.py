import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from typing import Union


class SAT25K(Dataset):
    def __init__(
            self, root_dir: Union[str, Path],
            image_size: int,
            image_channels: int,
            image_max_value: int,
            image_min_value: int,
            num_classes: int
    ):
        super().__init__()

        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.image_size = image_size
        self.image_channels = image_channels
        self.image_max_value = image_max_value
        self.image_min_value = image_min_value
        self.num_classes = num_classes

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory {self.root_dir} not found")

        self.image_dir = self.root_dir / 'images'
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Dataset image directory {self.image_dir} not found")

        self.annotation_dir = self.root_dir / 'annotations'
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Dataset annotation directory {self.annotation_dir} not found")

        self.image_files = sorted(list(self.image_dir.glob('**/*.npy')))
        self.annotation_files = sorted(list(self.annotation_dir.glob('**/*.npy')))

        assert len(self.image_files) == len(self.annotation_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_filename = self.image_files[index]
        annotation_filename = self.annotation_files[index]
        assert image_filename.stem == annotation_filename.stem
        image = self.process_image(np.load(str(image_filename)))
        annotation = self.process_annotation(np.load(str(annotation_filename)))
        return image, annotation

    def process_image(self, image: np.ndarray):
        assert image.shape == (self.image_channels, self.image_size, self.image_size)
        assert np.issubdtype(image.dtype, np.integer)
        assert np.all(image >= self.image_min_value) and np.all(image <= self.image_max_value)
        image = image.astype(np.float32)
        image = (image - self.image_min_value) / (self.image_max_value - self.image_min_value)
        image = image * 2.0 - 1.0
        assert np.all(image >= -1.0) and np.all(image <= 1.0)
        return image

    def process_annotation(self, annotation: np.ndarray):
        assert self.num_classes > 1
        assert annotation.shape == (self.image_size, self.image_size)
        assert np.issubdtype(annotation.dtype, np.integer)
        annotation_one_hot = np.zeros((self.num_classes, self.image_size, self.image_size))
        for i in range(self.num_classes):
            annotation_one_hot[i, :, :] = np.where(annotation == i, 1.0, 0.0)
        assert annotation_one_hot.shape == (self.num_classes, self.image_size, self.image_size)
        if self.num_classes == 2:
            annotation_one_hot = annotation_one_hot[self.num_classes - 1, ...][None, ...]
            assert annotation_one_hot.shape == (1, self.image_size, self.image_size)
        return annotation_one_hot


def load_sat25k(
        data_dir: Union[str, Path],
        batch_size: int,
        image_size: int,
        image_channels: int,
        image_max_value: int,
        image_min_value: int,
        num_classes: int,
        is_train: bool,
        shuffle: bool
):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    root_dir = data_dir / 'training' if is_train else 'validation'
    dataset = SAT25K(root_dir, image_size, image_channels, image_max_value, image_min_value, num_classes)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        drop_last=True
    )
    while True:
        yield from loader

