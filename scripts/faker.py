"""
Populates the data directory with fake data (.npy) for testing purposes only.

The idea is to generate one sample manually and then augment it to generate more test samples.

The nuance is that the faked samples and their corresponding annotations are actually alinged
rather than being independent and totally random.

NOTE: This script is not meant to be used in production. Only num_classes == 2 (binary) case is considered.

Directory Structure:

data_dir
├── training
│   ├── images
│   └── annotations
└── validation
    ├── images
    └── annotations
"""

import os
import sys
from pathlib import Path
import shutil
import numpy as np
import uuid
import albumentations as A
import cv2
import argparse

from syn10_diffusion.utils import parse_config


def generate(data_dir: str, n_samples: int, sample: np.ndarray, label: np.ndarray, is_train: bool):
    print(f"Generating {n_samples} samples for {'train' if is_train else 'validation'} ... ", end="")

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    split_dir = data_dir / "training" if is_train else data_dir / "validation"
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"

    if any([images_dir.exists(), annotations_dir.exists()]):
        shutil.rmtree(images_dir)
        shutil.rmtree(annotations_dir)

    os.makedirs(images_dir, exist_ok=False)
    os.makedirs(annotations_dir, exist_ok=False)

    augmentation = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT, value=0
            )
        ]
    )

    for _ in range(n_samples):
        augmented = augmentation(image=sample, mask=label)
        image = augmented['image']
        annotation = augmented['mask']

        name = str(uuid.uuid4().hex)
        suffix = ".npy"

        np.save(str(images_dir / (name + suffix)), image)
        np.save(str(annotations_dir / (name + suffix)), annotation)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Faker")
    parser.add_argument("--config", help="path to config file", type=str, required=True)
    parser.add_argument("--data_dir", help="path to data directory", type=str, required=True)
    parser_args = parser.parse_args()
    config = parse_config(parser_args.config)

    object_size = config['image_size'] // 2
    object_location = config['image_size'] // 2 - object_size // 2

    sample = np.zeros(shape=(config['image_channels'], config['image_size'], config['image_size']))
    sample[:, object_location:object_location+object_size, object_location:object_location+object_size] = \
        config['image_max_value']
    sample = sample.astype(np.uint8)

    label = np.zeros(shape=(config['image_size'], config['image_size']))
    label[object_location:object_location+object_size, object_location:object_location+object_size] = 1
    label = label.astype(np.uint8)

    generate(data_dir=parser_args.data_dir, n_samples=100, sample=sample, label=label, is_train=True,)
    generate(data_dir=parser_args.data_dir, n_samples=20, sample=sample, label=label, is_train=False)


if __name__ == "__main__":
    sys.exit(main())
