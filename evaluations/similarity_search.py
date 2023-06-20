import os
import sys
import json
import argparse
from typing import List
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torchvision import models
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from evaluations.dataset import load_ss_image_dataset, load_ss_search_dataset
from syn10_diffusion import utils

utils.seed_all()


@torch.no_grad()
def get_activations(model, dataset):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    hook_handle = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            hook_handle = module.register_forward_hook(hook_fn)

    model(dataset)

    if hook_handle is not None:
        hook_handle.remove()

    return torch.squeeze(activations[0], dim=(-2, -1))


def cosine_similarity(
        image_embeddings: torch.Tensor,
        search_embeddings: torch.Tensor
):
    assert image_embeddings.ndim == 2
    assert search_embeddings.ndim == 2
    assert image_embeddings.shape[1] == search_embeddings.shape[1]

    # Normalize the embeddings
    image_embeddings = F.normalize(image_embeddings, dim=1)  # (n, d)
    search_embeddings = F.normalize(search_embeddings, dim=1)  # (b, d)

    # Compute cosine similarity
    similarity_matrix = torch.mm(search_embeddings, image_embeddings.t())  # (b, n)

    return similarity_matrix


def inverse_transform(
        inp: torch.Tensor,
        image_size: int,
        image_max_value: int,
        image_min_value: int
):
    assert inp.ndim == 5  # (k, n, 3, 299, 299)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=inp.dtype, device=inp.device).reshape(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=inp.dtype, device=inp.device).reshape(1, 1, 3, 1, 1)

    inp = inp * std + mean
    inp = inp * (image_max_value - image_min_value) + image_min_value
    inp = torch.clamp(inp, min=image_min_value, max=image_max_value)

    inp_4d = inp.view(-1, *inp.shape[2:])  # (k*n, 3, 299, 299)
    out_4d = TF.resize(inp_4d, size=[image_size, image_size], interpolation=TF.InterpolationMode.NEAREST, antialias=True)
    out = out_4d.view(*inp.shape[:2], *out_4d.shape[1:])
    return out


@torch.no_grad()
def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def check_args(args):
    if not Path(args.search_path).exists():
        raise RuntimeError(f"Search path {args.search_path} not found")
    for p in args.image_path:
        if not Path(p).exists():
            raise RuntimeError(f"Image path {p} not found")
    if not Path(args.save_path).exists():
        raise RuntimeError(f"Save path {args.save_dir} not found")
    assert args.image_size > 0
    assert 0 < args.top_k <= args.batch_size
    assert args.image_max_value > args.image_min_value >= 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_path", help="where to search", type=str, required=True)
    parser.add_argument("--image_path", help="what to search for (list)", type=str, nargs="+", required=True)
    parser.add_argument("--top_k", help="most similar k search results", type=int, required=True)
    parser.add_argument("--batch_size", help="batch size", type=int, required=True)
    parser.add_argument("--image_size", help="image size", type=int, required=True)
    parser.add_argument("--image_max_value", help="maximum allowed pixel value", type=int, default=255)
    parser.add_argument("--image_min_value", help="minimum allowed pixel value", type=int, default=0)
    parser.add_argument("--save_path", help="path to save results", type=str, required=True)

    args = parser.parse_args()
    check_args(args)

    search_path: Path = Path(args.search_path)
    image_path: List[Path] = [Path(p) for p in args.image_path]
    save_path: Path = Path(args.save_path)

    dist.init_process_group(backend="nccl", init_method="env://")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_id = os.environ['TORCHELASTIC_RUN_ID']

    print(f"Rank {rank} is loading Inception_v3 model")
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
    sync_model(inception_v3)  # sync point

    print(f"Rank {rank} is loading samples")
    image_dataset = load_ss_image_dataset(
        image_path=image_path,
        image_max_value=args.image_max_value,
        image_min_value=args.image_min_value
    )  # (n, 3, 299, 299), all devices have the same image dataset

    image_dataset = image_dataset.to(local_rank)  # dtype=torch.float32, device=local_rank
    _, c, h, w = image_dataset.shape  # (b, 3, 299, 299) Inception v3 input shape

    print(f"Rank {rank} is computing sample embeddings")
    image_embeddings = get_activations(
        model=inception_v3,
        dataset=image_dataset
    )  # (n, d=2048), dtype=torch.float32, device=local_rank

    print(f"Rank {rank} is loading search dataset")
    search_dataset_loader = load_ss_search_dataset(
        search_path=search_path,
        batch_size=args.batch_size,
        image_max_value=args.image_max_value,
        image_min_value=args.image_min_value
    )

    storage = []
    for i, search_dataset in enumerate(search_dataset_loader):
        print(f"Rank: {rank}, search dataset batch_idx: {i+1}/{len(search_dataset_loader)}")
        search_dataset = search_dataset.to(local_rank)  # (b, 3, 299, 299), dtype=torch.float32, device=local_rank
        print(f"Rank {rank} is computing batch embeddings")
        search_embeddings = get_activations(
            model=inception_v3,
            dataset=search_dataset
        )

        print(f"Rank {rank} is computing similarity matrix")
        similarity_matrix = cosine_similarity(
            image_embeddings=image_embeddings,
            search_embeddings=search_embeddings
        )  # (search_dataset.shape[0], image_embeddings.shape[0]), (b, n)

        top_k_indices = torch.topk(similarity_matrix, k=args.top_k, dim=0).indices  # (k, n)
        top_k_images = search_dataset[top_k_indices]  # (k, n, 3, 299, 299)
        top_k_similarities = torch.gather(similarity_matrix, dim=0, index=top_k_indices)  # (k, n)
        storage.extend([top_k_images, top_k_similarities])
        if i > 0:
            temp_images = torch.cat([storage[0], top_k_images], dim=0)  # (2k, n, 3, 299, 299)
            temp_similarities = torch.cat([storage[1], top_k_similarities], dim=0)  # (2k, n)
            temp_indices = torch.topk(temp_similarities, k=args.top_k, dim=0).indices  # (k, n)
            temp_sliced_images = torch.gather(
                temp_images,
                dim=0,
                index=temp_indices.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                .expand(-1, -1, c, h, w)
            )  # (k, n, 3, 299, 299)
            temp_scores = torch.gather(temp_similarities, dim=0, index=temp_indices)  # (k, n)
            storage = [temp_sliced_images, temp_scores]

    top_k_images = storage[0].contiguous()
    top_k_similarities = storage[1].contiguous()

    gathered_top_k_images = [torch.zeros_like(top_k_images) for _ in range(world_size)]
    dist.all_gather(gathered_top_k_images, top_k_images)  # sync point

    gathered_top_k_similarities = [torch.zeros_like(top_k_similarities) for _ in range(world_size)]
    dist.all_gather(gathered_top_k_similarities, top_k_similarities)  # sync point

    gathered_top_k_images = torch.cat(gathered_top_k_images, dim=0)  # (k * world_size, n, 3, 299, 299)
    gathered_top_k_similarities = torch.cat(gathered_top_k_similarities, dim=0)  # (k * world_size, n)
    top_k_indices = torch.topk(gathered_top_k_similarities, k=args.top_k, dim=0).indices  # (k, n)
    top_k_images = torch.gather(
        gathered_top_k_images,
        dim=0,
        index=top_k_indices.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        .expand(-1, -1, c, h, w)
    )  # (k, n, 3, 299, 299)
    top_k_similarities = torch.gather(gathered_top_k_similarities, dim=0, index=top_k_indices)  # (k, n)

    if rank == 0:
        top_k_images = inverse_transform(
            top_k_images,
            args.image_size,
            args.image_max_value,
            args.image_min_value
        )  # (k, n, 3, args.image_size, args.image_size)
        top_k_images = top_k_images.cpu().numpy()
        top_k_similarities = top_k_similarities.cpu().numpy()
        save_path = save_path / "similarity"
        save_path.mkdir(parents=True, exist_ok=True)
        top_k_images_path = str(save_path / f"top_k_images_{run_id}.npy")
        top_k_similarities_path = str(save_path / f"top_k_similarities_{run_id}.npy")
        np.save(top_k_images_path, top_k_images.astype(np.uint16))
        np.save(top_k_similarities_path, top_k_similarities.astype(np.float32))
        with open(save_path / f"top_k_{run_id}.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        print(f"Saved top k images to {top_k_images_path}")
        print(f"Saved top k similarities to {top_k_similarities_path}")
        print("Done!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
