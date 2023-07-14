import sys
import json
import uuid
from pathlib import Path
import argparse
import numpy as np
import torch
import torchvision.transforms.functional as TF

from syn10_diffusion import utils
from syn10_diffusion.models import UnetProd
from syn10_diffusion.diffusion import Diffusion

utils.seed_all()


def load_image(
        image_path: Path,
        resize: int,
        image_max_value: int,
        image_min_value: int
):
    image: np.ndarray = np.load(str(image_path))
    assert image.ndim == 3, f"Image must be 3D, got {image.ndim}"
    assert image.shape[0] in [1, 3], f"Image must have 1 or 3 channels (channel first), got {image.shape[0]}"
    assert image.shape[1] == image.shape[2], f"Image must be square, got {image.shape[1]}x{image.shape[2]}"
    assert np.issubdtype(image.dtype, np.integer), f"Image must be integer type, got {image.dtype}"
    assert np.all(image >= image_min_value) and np.all(image <= image_max_value)

    image = torch.from_numpy(image.astype(np.float32))
    if image.shape[1] != resize:
        image = TF.resize(
            image,
            size=[resize, resize],
            interpolation=TF.InterpolationMode.NEAREST,
            antialias=True
        )

    image = (image - image_min_value) / (image_max_value - image_min_value)
    image = image * 2.0 - 1.0
    assert torch.all(image >= -1.0) and torch.all(image <= 1.0)
    return image.unsqueeze(0)


def load_mask(
        mask_path: Path,
        resize: int,
        num_classes: int
):
    mask: np.ndarray = np.load(str(mask_path))
    assert mask.ndim == 2, f"Mask must be 2D, got {mask.ndim}"
    assert mask.shape[0] == mask.shape[1], f"Mask must be square, got {mask.shape[0]}x{mask.shape[1]}"
    assert np.issubdtype(mask.dtype, np.integer), f"Mask must be integer type, got {mask.dtype}"
    assert np.all(mask >= 0) and np.all(mask < num_classes), \
        f"Mask must be in range [0, {num_classes}), got {mask.min()} and {mask.max()}"

    mask_size = mask.shape[0]
    mask_one_hot = np.zeros((num_classes, mask_size, mask_size), dtype=np.float32)
    for i in range(num_classes):
        mask_one_hot[i, :, :] = np.where(mask == i, 1.0, 0.0)

    mask_one_hot = torch.from_numpy(mask_one_hot)
    if mask_size != resize:
        mask_one_hot = TF.resize(
            mask_one_hot,
            size=[resize, resize],
            interpolation=TF.InterpolationMode.NEAREST,
            antialias=True
        )
    return mask_one_hot.unsqueeze(0)


def validate_args(args):
    if not Path(args.image_a_path).exists():
        raise RuntimeError(f"Image file {args.image_a_path} not found")
    if not Path(args.image_b_path).exists():
        raise RuntimeError(f"Image file {args.image_b_path} not found")
    if not Path(args.mask_path).exists():
        raise RuntimeError(f"Mask file {args.mask_path} not found")
    if not Path(args.model_path).exists():
        raise RuntimeError(f"Model file {args.model_path} not found")
    if not Path(args.config).exists():
        raise RuntimeError(f"Config file {args.config} not found")
    if not Path(args.save_path).exists():
        raise RuntimeError(f"Save path {args.save_path} not found")
    assert args.num_steps > 0, "Number of steps must be greater than 0"
    assert all([0 < lamda < 1 for lamda in args.lambda_interpolate]), \
        "Interpolation factor must be between 0 and 1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_a_path", help="path to the first", type=str, required=True)
    parser.add_argument("--image_b_path", help="path to the second image", type=str, required=True)
    parser.add_argument("--mask_path", help="path to mask file", type=str, required=True)
    parser.add_argument("--model_path", help="path to model file", type=str, required=True)
    parser.add_argument("--config", help="path to model config file", type=str, required=True)
    parser.add_argument("--num_steps", help="number of steps during interpolation", type=int, required=True)
    parser.add_argument("--lambda_interpolate", help="interpolation factor (0.0, 1.0)", type=float, nargs="+", required=True)
    parser.add_argument("--save_path", help="path to save results", type=str, required=True)

    parser_args = parser.parse_args()
    validate_args(parser_args)
    config = utils.parse_config(parser_args.config)
    utils.validate_config(config)

    image_a_path: Path = Path(parser_args.image_a_path)
    image_b_path: Path = Path(parser_args.image_b_path)
    mask_path: Path = Path(parser_args.mask_path)
    model_path: Path = Path(parser_args.model_path)
    save_path: Path = Path(parser_args.save_path)
    num_steps = parser_args.num_steps

    image_size = config['image_size']
    image_max_value = config['image_max_value']
    image_min_value = config['image_min_value']
    num_classes = config['num_classes']

    run_id = str(uuid.uuid4())
    save_path = save_path / "interpolation"
    save_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading encoder...")
    diffusion = Diffusion(**config)

    print("Loading decoder...")
    model = UnetProd(**config).to(device)
    model.eval()
    model_checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(model_checkpoint['model_state_dict'])

    print("Loading images A and B...")
    image_a: torch.Tensor = load_image(
        image_a_path,
        image_size,
        image_max_value,
        image_min_value
    )  # (1, C, H, W), [-1.0, 1.0], torch.float32
    image_a = image_a.to(device)

    image_b: torch.Tensor = load_image(
        image_b_path,
        image_size,
        image_max_value,
        image_min_value
    )  # (1, C, H, W), [-1.0, 1.0], torch.float32
    image_b = image_b.to(device)

    print("Loading mask...")
    mask: torch.Tensor = load_mask(
        mask_path,
        image_size,
        num_classes
    )  # (1, NC, H, W), [0.0, 1.0], torch.float32
    mask = mask.to(device)

    t = torch.tensor([num_steps]).long().to(device)
    for lamda in parser_args.lambda_interpolate:
        print(f"Interpolating with lambda={lamda}...")
        with torch.no_grad():
            noisy_image_a = diffusion.q_sample(image_a, t)
            noisy_image_b = diffusion.q_sample(image_b, t)
            noisy_image_interpolated = (1 - lamda) * noisy_image_a + lamda * noisy_image_b
            sample = diffusion.p_sample_interpolate(
                model,
                noisy_image_a.shape,
                model_kwargs={
                    'x': noisy_image_interpolated,
                    'y': mask,
                    'num_steps': num_steps,
                    'guidance': config['guidance'],
                    'model_output_channels': config['model_output_channels']
                }
            )
            sample = (sample + 1.0) / 2.0 * (config['image_max_value'] - config['image_min_value']) \
                + config['image_min_value']
            sample = sample.clamp(config['image_min_value'], config['image_max_value'])  # (1, C, H, W)
            sample = sample.squeeze(0).cpu().numpy()

            file_path = save_path / f"interpolate_{lamda}_{run_id}.npy"
            np.save(str(file_path), sample.astype(np.uint16))
            print(f"Saved to {file_path}")

    payload = vars(parser_args)
    file_path = save_path / f"interpolate_{run_id}.json"
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    sys.exit(main())
