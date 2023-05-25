"""
Populates the data directory with fake data (.npy) for testing purposes only.
The idea is to generate one sample manually and then augment it to generate more test samples.
Faked samples and their corresponding annotations are alinged.

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


def generate(
        sample: np.ndarray,
        label: np.ndarray,
        n_samples: int,
        image_max_value: int,
        image_min_value: int,
        save_dir: str,
        is_train: bool
):
    print(f"Generating {n_samples} samples for {'train' if is_train else 'validation'} ... ", end="")

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    split_dir = save_dir / "training" if is_train else save_dir / "validation"
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
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            )
        ]
    )

    for _ in range(n_samples):
        augmented = augmentation(image=sample, mask=label)
        image = augmented['image'].clip(image_min_value, image_max_value).transpose((2, 0, 1)).astype(np.uint16)
        annotation = augmented['mask'].clip(0, 1).astype(np.uint16)

        name = str(uuid.uuid4().hex)
        suffix = ".npy"

        np.save(str(images_dir / (name + suffix)), image)
        np.save(str(annotations_dir / (name + suffix)), annotation)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Faker")
    parser.add_argument("--image_size", help="image size", type=int, required=True)
    parser.add_argument("--image_channels", help="number of image channels", type=int, required=True)
    parser.add_argument("--image_max_value", help="maximum value of image", type=int, required=True)
    parser.add_argument("--image_min_value", help="minimum value of image", type=int, required=True)
    parser.add_argument("--num_train_samples", help="number of training samples", type=int, required=True)
    parser.add_argument("--num_val_samples", help="number of validation samples", type=int, required=True)
    parser.add_argument("--save_dir", help="path to data directory", type=str, required=True)
    parser_args = parser.parse_args()

    object_size = parser_args.image_size // 2
    object_location = parser_args.image_size // 2 - object_size // 2

    sample = np.zeros(shape=(parser_args.image_size, parser_args.image_size, parser_args.image_channels))
    sample[object_location:object_location+object_size, object_location:object_location+object_size, :] = \
        parser_args.image_max_value

    label = np.zeros(shape=(parser_args.image_size, parser_args.image_size))
    label[object_location:object_location+object_size, object_location:object_location+object_size] = 1

    generate(
        sample=sample,
        label=label,
        n_samples=parser_args.num_train_samples,
        image_max_value=parser_args.image_max_value,
        image_min_value=parser_args.image_min_value,
        save_dir=parser_args.save_dir,
        is_train=True
    )

    generate(
        sample=sample,
        label=label,
        n_samples=parser_args.num_val_samples,
        image_max_value=parser_args.image_max_value,
        image_min_value=parser_args.image_min_value,
        save_dir=parser_args.save_dir,
        is_train=False
    )


if __name__ == "__main__":
    sys.exit(main())
