"""Generates fake data (.npy) for testing purposes.

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


def generate(data_dir="/home/orkhan/sat25k", n_samples=100, is_train=True, **params):
    print(f"generating {n_samples} samples for {'train' if is_train else 'validation'} ... ", end="")

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

    for _ in range(n_samples):
        image = np.random.randint(
            params['image_min_value'],
            params['image_max_value'] + 1,
            size=(
                params['image_channels'],
                params['image_size'],
                params['image_size']
            )
        )
        annotation = np.random.randint(
            0,
            params['num_classes'],
            size=(
                params['image_size'],
                params['image_size']
            )
        )
        name = str(uuid.uuid4().hex)
        suffix = ".npy"
        np.save(str(images_dir / (name + suffix)), image)
        np.save(str(annotations_dir / (name + suffix)), annotation)

    print("Done.")


def main():

    params = {
        "image_channels": 3,
        "image_size": 128,
        "image_max_value": 255,
        "image_min_value": 0,
        "num_classes": 2
    }

    generate(data_dir="/home/orkhan/sat25k", n_samples=100, is_train=True, **params)
    generate(data_dir="/home/orkhan/sat25k", n_samples=20, is_train=False, **params)


if __name__ == "__main__":
    sys.exit(main())
