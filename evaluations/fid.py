import os
from pathlib import Path
import argparse
import warnings
import numpy as np
from scipy import linalg
import torch
import torch.distributed as dist
from torchvision import models
from evaluations.dataset import load_fid_dataset
from syn10_diffusion import utils

utils.seed_all()


def check_args(args):
    if not Path(args.real_path).exists():
        raise RuntimeError(f"Real path {args.real_path} not found")
    if not Path(args.syn_path).exists():
        raise RuntimeError(f"Syn path {args.syn_path} not found")
    if not Path(args.save_path).exists():
        raise RuntimeError(f"Save path {args.save_dir} not found")
    assert args.batch_size > 0
    assert args.image_max_value > args.image_min_value >= 0


@torch.no_grad()
def get_activations(model, data_loader, world_size):
    activations = []
    all_activations = []
    device = next(model.parameters()).device

    def hook_fn(module, input, output):
        activations.append(output)

    hook_handle = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            hook_handle = module.register_forward_hook(hook_fn)

    for i, batch in enumerate(data_loader):
        print(
            f"Rank: {dist.get_rank()}, "
            f"batch_size: {batch.shape[0]}, "
            f"i_batch: {i + 1}/{len(data_loader)}"
        )
        batch = batch.to(device)
        model(batch)

    concat_activations = torch.cat(activations, dim=0)
    gathered_activations = [torch.zeros_like(concat_activations) for _ in range(world_size)]
    concat_activations.contiguous()
    dist.all_gather(gathered_activations, concat_activations)
    all_activations.extend([activation.cpu().numpy() for activation in gathered_activations])
    if hook_handle is not None:
        hook_handle.remove()
    return all_activations


def calculate_fid(real_activations: np.ndarray, syn_activations: np.ndarray, eps: float = 1e-6):
    """
    Compute the Frechet distance between two sets of statistics.
    """
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132

    assert real_activations.ndim == syn_activations.ndim == 2
    assert real_activations.shape[1] == syn_activations.shape[1] == 2048

    mu1, sigma1 = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = np.mean(syn_activations, axis=0), np.cov(syn_activations, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
            mu1.shape == mu2.shape
    ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
    assert (
            sigma1.shape == sigma2.shape
    ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def save(fid_score, save_path):
    file = Path(save_path) / f"fid_score_{os.environ['TORCHELASTIC_RUN_ID']}.txt"
    with open(file, "w") as f:
        f.write(str(fid_score))
    print(f"Saved FID score to {file}")


@torch.no_grad()
def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", help="path to real dataset ", type=str, required=True)
    parser.add_argument("--syn_path", help="path to synthetic dataset", type=str, required=True)
    parser.add_argument("--batch_size", help="batch size", type=int, default=4)
    parser.add_argument("--image_max_value", help="maximum allowed pixel value", type=int, default=255)
    parser.add_argument("--image_min_value", help="minimum allowed pixel value", type=int, default=0)
    parser.add_argument("--save_path", help="path to save results", type=str, required=True)

    args = parser.parse_args()
    check_args(args)

    real_path = Path(args.real_path)
    syn_path = Path(args.syn_path)
    batch_size = args.batch_size

    dist.init_process_group(backend="nccl", init_method="env://")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        inception_v3 = models.inception_v3(
            weights="IMAGENET1K_V1",
            init_weights=False,
            transform_input=False
        ).to(local_rank)
    else:
        inception_v3 = models.inception_v3(
            weights=None,
            init_weights=True,
            transform_input=False
        ).to(local_rank)

    inception_v3.eval()
    sync_model(inception_v3)

    real_dataloader = load_fid_dataset(
        image_dir=real_path,
        batch_size=batch_size,
        image_max_value=args.image_max_value,
        image_min_value=args.image_min_value,
        shuffle=False,
        drop_last=False
    )

    syn_dataloader = load_fid_dataset(
        image_dir=syn_path,
        batch_size=batch_size,
        image_max_value=args.image_max_value,
        image_min_value=args.image_min_value,
        shuffle=False,
        drop_last=False
    )

    real_activations = get_activations(inception_v3, real_dataloader, world_size)
    syn_activations = get_activations(inception_v3, syn_dataloader, world_size)

    if rank == 0:
        real_activations = np.concatenate(real_activations, axis=0).squeeze((2, 3))
        syn_activations = np.concatenate(syn_activations, axis=0).squeeze((2, 3))
        fid_score = calculate_fid(real_activations, syn_activations)
        print(f"FID score: {fid_score:.4f}")
        save(fid_score, args.save_path)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
