"""
Output Directory Structure:

root
├── training
│   ├── images
│   └── annotations
└── validation
    ├── images
    └── annotations
"""

import uuid
import math
import json
from tqdm import tqdm
from pathlib import Path
import argparse
import cv2
import numpy as np
import albumentations as A
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import Window, bounds
from rasterio.mask import raster_geometry_mask
from typing import Union, List, Tuple

from syn10_diffusion import utils

utils.seed_all()


class Tiler:
    def __init__(
            self,
            image_root_dir: Union[str, Path],
            image_extensions: List[str],
            mask_root_dir: Union[str, Path],
            mask_extensions: List[str],
            tile_size: int,
            tile_overlap: float,
            tile_keep_ratio: float,
            downsample_factor: int,
            image_max_value: int,
            image_min_value: int,
            image_channels: int,
            num_train_samples: int,
            num_val_samples: int,
            save_dir: Union[str, Path]

    ):
        self.image_root_dir = Path(image_root_dir) if isinstance(image_root_dir, str) else image_root_dir
        self.image_extensions = image_extensions
        self.mask_root_dir = Path(mask_root_dir) if isinstance(mask_root_dir, str) else mask_root_dir
        self.mask_extensions = mask_extensions
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_keep_ratio = tile_keep_ratio
        self.downsample_factor = downsample_factor
        self.image_max_value = image_max_value
        self.image_min_value = image_min_value
        self.image_channels = image_channels
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_total_samples = self.num_train_samples + self.num_val_samples
        self.save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
        self.image_files: List[Path] = self.load_image_files(self.image_root_dir)
        self.mask_files: List[Path] = self.load_mask_files(self.mask_root_dir)
        self.image_mask_correspondence: List[Tuple] = self.make_correspondence(self.image_files, self.mask_files)

    @staticmethod
    def norm_to_uint8(norm_image: np.ndarray) -> np.ndarray:
        assert norm_image.dtype == np.float32
        assert np.all(norm_image >= 0.0) and np.all(norm_image <= 1.0)
        image_uint8 = (norm_image * 255).astype(np.uint8)
        return image_uint8

    @staticmethod
    def uint8_to_norm(image: np.ndarray) -> np.ndarray:
        assert image.dtype == np.uint8
        assert np.all(image >= 0) and np.all(image <= 255)
        norm_image = image.astype(np.float32) / 255
        return norm_image

    @staticmethod
    def make_correspondence(image_files: List[Path], mask_files: List[Path]) -> List[Tuple]:
        image_stems = [image_file.stem for image_file in image_files]
        mask_stems = [mask_file.stem for mask_file in mask_files]
        assert len(image_stems) == len(mask_stems), "Number of image files and mask files do not match."
        image_set = set(image_stems)
        assert len(image_set) == len(image_stems), "Image file names are not unique."
        mask_set = set(mask_stems)
        assert len(mask_set) == len(mask_stems), "Mask file names are not unique."
        assert image_set == mask_set, "Image and mask files do not match."
        image_mask_correspondence = [
            (image_file, mask_files[mask_stems.index(image_file.stem)]) for image_file in image_files
        ]
        return image_mask_correspondence

    @staticmethod
    def augment(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert image.dtype == np.uint8 and np.all(image >= 0) and np.all(image <= 255)
        assert mask.dtype == np.uint8 and np.all(mask >= 0) and np.all(mask <= 255)
        image = image.transpose(1, 2, 0)

        stochastic_pipe = A.Compose([
            A.RandomRotate90(p=1.0),
            A.VerticalFlip(p=0.5),
        ])

        deterministic_pipe = A.Compose([
            A.VerticalFlip(p=1.0),
        ])

        augmented = stochastic_pipe(image=image, mask=mask)
        image_aug = augmented['image']
        mask_aug = augmented['mask']

        if np.array_equal(image_aug, image) or np.array_equal(mask_aug, mask):
            augmented = deterministic_pipe(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = augmented['mask']

        assert np.array_equal(image_aug, image) is False, "Augmentation failed."
        assert np.array_equal(mask_aug, mask) is False, "Augmentation failed."
        image_aug = image_aug.transpose(2, 0, 1)
        return image_aug, mask_aug

    def load_image_files(self, image_root_dir: Path) -> List[Path]:
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(list(image_root_dir.rglob(f"*.{ext}")))
        assert len(image_files) > 0, f"No image files found in {image_root_dir}."
        return image_files

    def load_mask_files(self, mask_root_dir: Path) -> List[Path]:
        mask_files = []
        for ext in self.mask_extensions:
            mask_files.extend(list(mask_root_dir.rglob(f"*.{ext}")))
        assert len(mask_files) > 0, f"No mask files found in {mask_root_dir}."
        return mask_files

    def norm(self, image: np.ndarray) -> np.ndarray:
        assert np.issubdtype(image.dtype, np.integer)
        image_norm = (image.astype(np.float32) - self.image_min_value) / (self.image_max_value - self.image_min_value)
        return image_norm

    def denorm(self, norm_image: np.ndarray) -> np.ndarray:
        assert norm_image.dtype == np.float32
        assert np.all(norm_image >= 0.0) and np.all(norm_image <= 1.0)
        image = norm_image * (self.image_max_value - self.image_min_value) + self.image_min_value
        image = np.clip(image, self.image_min_value, self.image_max_value).astype(np.uint16)
        return image

    def downsample(self, image, mask):
        assert image.dtype == np.uint8 and np.all(image >= 0) and np.all(image <= 255)
        assert mask.dtype == np.uint8 and np.all(mask >= 0) and np.all(mask <= 255)
        image = image.transpose(1, 2, 0)
        new_h = image.shape[0] // self.downsample_factor
        new_w = image.shape[1] // self.downsample_factor
        downsampled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        downsampled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        downsampled_image = downsampled_image.transpose(2, 0, 1)
        return downsampled_image, downsampled_mask

    def save(self, image, mask, for_train=True):
        save_dir = self.save_dir / "training" if for_train else self.save_dir / "validation"
        image_dir = save_dir / "images"
        mask_dir = save_dir / "annotations"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        filename = str(uuid.uuid4().hex)
        image_path = image_dir / f"{filename}.npy"
        mask_path = mask_dir / f"{filename}.npy"
        np.save(image_path, image)
        np.save(mask_path, mask)

    def run(self):
        tile_counter = 0
        for image_file, mask_file in self.image_mask_correspondence:
            print(f"Processing {image_file.name}...")
            with rasterio.open(image_file) as dataset:
                h, w = dataset.shape

                window_x_start = 0
                window_y_start = 0
                window_x_end = self.tile_size
                window_y_end = self.tile_size
                offset = math.ceil(self.tile_size * (1.0 - self.tile_overlap))

                while window_y_end <= h:
                    while window_x_end <= w:
                        window = Window.from_slices((window_y_start, window_y_end), (window_x_start, window_x_end))
                        window_transform = dataset.window_transform(window)
                        left, bottom, right, top = bounds(window, dataset.transform)
                        window_geometry = box(left, bottom, right, top)
                        window_mask = gpd.GeoDataFrame(geometry=[window_geometry], crs=dataset.crs)

                        mask_gdf = gpd.read_file(mask_file, mask=window_mask, engine="fiona")
                        mask_gdf = mask_gdf.to_crs(dataset.crs)
                        if mask_gdf.empty:
                            print(f"Skipping tile...", end="\r")
                            window_x_start += offset
                            window_x_end += offset
                            continue

                        image = dataset.read(window=window)
                        assert np.issubdtype(image.dtype, np.integer), "Image dtype must be integer."
                        assert image.ndim == 3, "Image must be 3-dimensional."

                        with rasterio.MemoryFile() as memfile:
                            with memfile.open(
                                    driver="GTiff",
                                    height=image.shape[1],
                                    width=image.shape[2],
                                    count=image.shape[0],
                                    dtype=image.dtype,
                                    transform=window_transform,
                                    crs=dataset.crs,
                            ) as mem_dataset:
                                mem_dataset.write(image)
                                masked, _, _ = raster_geometry_mask(
                                    mem_dataset, mask_gdf.geometry, invert=True
                                )
                                assert masked.ndim == 2, "Masked image must be 2-dimensional."
                                assert masked.dtype == bool, "Masked image must be boolean."
                                assert masked.shape == (image.shape[1], image.shape[2])

                                if self.downsample_factor > 1:
                                    image, masked = self.downsample(
                                        self.norm_to_uint8(self.norm(image)),
                                        self.norm_to_uint8(masked.astype(np.float32))
                                    )  # np.uint8, np.uint8

                                    image = self.denorm(self.uint8_to_norm(image))  # np.uint16
                                    masked = self.uint8_to_norm(masked).astype(np.uint16)  # np.uint16

                                if masked.sum() / masked.size < self.tile_keep_ratio:
                                    print(f"Dropping tile...", end="\r")
                                    window_x_start += offset
                                    window_x_end += offset
                                    continue

                                tile_counter += 1
                                print(f"Saving tile...", end="\r")
                                self.save(
                                    image.astype(np.uint16),
                                    masked.astype(np.uint16),
                                    for_train=tile_counter > self.num_val_samples
                                )

                                if tile_counter == self.num_total_samples:
                                    break

                        window_x_start += offset
                        window_x_end += offset

                    if tile_counter == self.num_total_samples:
                        break

                    window_x_start = 0
                    window_x_end = self.tile_size
                    window_y_start += offset
                    window_y_end += offset

            if tile_counter == self.num_total_samples:
                break

        assert tile_counter > 0, "No tiles were created."

        if tile_counter <= self.num_val_samples:
            summary = {
                "Validation tiles [organic]": tile_counter,
                "Training tiles [organic]": 0,
                "Training tiles [augmented]": 0,
                "Status": "Not enough training tiles were created."
            }
        else:
            if tile_counter == self.num_total_samples:
                summary = {
                    "Validation tiles [organic]": self.num_val_samples,
                    "Training tiles [organic]": self.num_train_samples,
                    "Training tiles [augmented]": 0,
                    "Status": "Completed"
                }
            else:
                num_train_samples_augmented = self.num_total_samples - tile_counter
                saved_image_files = sorted(list(self.save_dir.glob("training/images/*.npy")))
                saved_mask_files = sorted(list(self.save_dir.glob("training/annotations/*.npy")))
                assert len(saved_image_files) == len(saved_mask_files), "Image and mask files do not match."
                assert set([saved_image_file.stem for saved_image_file in saved_image_files]) == \
                       set([saved_mask_file.stem for saved_mask_file in saved_mask_files]), \
                       "Image and mask files do not match."

                random_indices = np.random.choice(len(saved_image_files), num_train_samples_augmented, replace=True)
                for random_index in tqdm(random_indices, desc="Augmentation"):
                    image = np.load(saved_image_files[random_index])  # np.uint16
                    masked = np.load(saved_mask_files[random_index])  # np.uint16
                    image = self.norm_to_uint8(self.norm(image))  # np.uint8
                    masked = self.norm_to_uint8(masked.astype(np.float32))  # np.uint8
                    image_aug, masked_aug = self.augment(image, masked)
                    image_aug = self.denorm(self.uint8_to_norm(image_aug))  # np.uint16
                    masked_aug = self.uint8_to_norm(masked_aug).astype(np.uint16)  # np.uint16
                    self.save(image_aug, masked_aug, for_train=True)

                summary = {
                    "Validation tiles [organic]": self.num_val_samples,
                    "Training tiles [organic]": tile_counter - self.num_val_samples,
                    "Training tiles [augmented]": num_train_samples_augmented,
                    "Status": "Completed"
                }

        with open(self.save_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)

        print("-" * 80)
        print(summary)
        print("-" * 80)
        print(f"Saved metadata to {self.save_dir / 'summary.json'}")
        print(f"Saved data to {self.save_dir}")
        print("-" * 80)
        print(f"Done.")


def validate_args(args):
    if not Path(args.image_root_dir).exists():
        raise RuntimeError(f"Image root directory {args.image_root_dir} does not exist.")
    if not Path(args.mask_root_dir).exists():
        raise RuntimeError(f"Mask root directory {args.mask_root_dir} does not exist.")
    if Path(args.save_dir).exists():
        raise RuntimeError(f"Save directory {args.save_dir} already exists.")
    assert args.tile_size > 0, "Tile size must be greater than 0."
    assert args.tile_size & (args.tile_size - 1) == 0, "Tile size must be divisible by 2**n."
    assert 0.0 <= args.tile_overlap < 1.0, "Tile overlap must be in range [0.0, 1.0)."
    assert 0.0 < args.tile_keep_ratio <= 1.0, "Tile keep ratio must be in range (0.0, 1.0]."
    assert args.downsample_factor > 0, "Downsample must be greater than 0."
    assert args.downsample_factor & (args.downsample_factor - 1) == 0, "Downsample factor must be divisible by 2**n."
    assert args.tile_size % args.downsample_factor == 0, "Tile size must be divisible by downsample factor."
    assert args.tile_size // args.downsample_factor >= 1, "Tile size must be >= downsample factor."
    assert args.image_max_value > args.image_min_value >= 0, "Image max value must be greater than image min value."
    assert args.image_channels > 0, "Image channels must be greater than 0."
    assert args.num_train_samples > 0, "Number of training samples must be greater than 0."
    assert args.num_val_samples > 0, "Number of validation samples must be greater than 0."
    assert all(isinstance(ext, str) for ext in args.image_extensions), "Image extensions must be of type List[str]."
    assert all(isinstance(ext, str) for ext in args.mask_extensions), "Mask extensions must be of type List[str]."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root_dir", help="root level path to image files", type=str, required=True)
    parser.add_argument("--image_extensions", help="image file extensions", type=str, nargs="+", default=["tif"])
    parser.add_argument("--mask_root_dir", help="root level path to mask files", type=str, required=True)
    parser.add_argument("--mask_extensions", help="mask file extensions", type=str, nargs="+", default=["shp"])
    parser.add_argument("--tile_size", help="tile size", type=int, required=True)
    parser.add_argument("--tile_overlap", help="overlap between tiles [0.0, 1.0)", type=float, required=True)
    parser.add_argument("--tile_keep_ratio", help="ratio of positive class", type=float, required=True)
    parser.add_argument("--downsample_factor", help="downsample factor", type=int, required=True)
    parser.add_argument("--image_max_value", help="maximum pixel value of input image", type=int, required=True)
    parser.add_argument("--image_min_value", help="minimum pixel value of input image", type=int, required=True)
    parser.add_argument("--image_channels", help="number of image channels", type=int, required=True)
    parser.add_argument("--num_train_samples", help="number of training samples", type=int, required=True)
    parser.add_argument("--num_val_samples", help="number of validation samples", type=int, required=True)
    parser.add_argument("--save_dir", help="path to save directory", type=str, required=True)
    parser_args = parser.parse_args()

    validate_args(parser_args)

    tiler = Tiler(
        image_root_dir=parser_args.image_root_dir,
        image_extensions=parser_args.image_extensions,
        mask_root_dir=parser_args.mask_root_dir,
        mask_extensions=parser_args.mask_extensions,
        tile_size=parser_args.tile_size,
        tile_overlap=parser_args.tile_overlap,
        tile_keep_ratio=parser_args.tile_keep_ratio,
        downsample_factor=parser_args.downsample_factor,
        image_max_value=parser_args.image_max_value,
        image_min_value=parser_args.image_min_value,
        image_channels=parser_args.image_channels,
        num_train_samples=parser_args.num_train_samples,
        num_val_samples=parser_args.num_val_samples,
        save_dir=parser_args.save_dir,
    )

    tiler.run()


if __name__ == "__main__":
    main()
