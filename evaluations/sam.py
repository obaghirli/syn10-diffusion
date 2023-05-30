import os
from pathlib import Path
import argparse

import numpy as np
import cv2

import torch
import torch.distributed as dist
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from evaluations.dataset import load_sam_dataset
from syn10_diffusion import utils
from typing import Dict, List

utils.seed_all()


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


def get_point_coords_and_labels(annotation, num_points):
    num_classes = annotation.shape[2]
    point_coords = []
    point_labels = []

    for c in range(num_classes):
        (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(annotation[:, :, c])
        for component_id in range(num_labels):
            if component_id == 0:
                continue
            component_indices = np.stack(np.where(labels == component_id), axis=1)
            component_indices[:, [0, 1]] = component_indices[:, [1, 0]]
            sampling_idx = np.random.randint(0, component_indices.shape[0], size=num_points)
            point_coords.append(component_indices[sampling_idx])
            point_labels.append([0 if c == 0 else 1 for _ in range(num_points)])

    point_coords = np.stack(point_coords, axis=0)
    point_labels = np.stack(point_labels, axis=0)
    return point_coords, point_labels


def convert_to_sam_batch(images: np.ndarray, annotations: np.ndarray, num_points, encoder_img_size, device):
    batched_input: List[Dict] = []
    resize_transform = ResizeLongestSide(encoder_img_size)
    for image, annotation in zip(images, annotations):
        point_coords, point_labels = get_point_coords_and_labels(annotation, num_points)
        point_coords = torch.from_numpy(point_coords).to(device)
        point_labels = torch.from_numpy(point_labels).to(device)
        batched_input.append(
            {
                "image": prepare_image(image, resize_transform, device),
                "point_coords": resize_transform.apply_coords_torch(point_coords, image.shape[:2]),
                "point_labels": point_labels,
                "original_size": image.shape[:2],
            }
        )
    return batched_input


def calculate_batch_iou_score(pred_mask, true_mask):
    eps = 1e-6
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    iou = (intersection + eps) / (union + eps)
    return torch.clamp(iou, 0.0, 1.0)


def check_args(args):
    assert args.num_points > 0
    assert args.image_max_value > args.image_min_value >= 0
    assert args.image_size > 0
    assert args.batch_size > 0
    assert args.num_classes == 2, "Only binary segmentation is supported"
    assert Path(args.image_dir).exists()
    assert Path(args.annotation_dir).exists()
    assert Path(args.sam_checkpoint).exists()
    assert args.model_type in sam_model_registry.keys()
    assert Path(args.save_dir).exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", help="number of point prompts", type=int, required=True)
    parser.add_argument("--image_max_value", help="maximum allowed pixel value", type=int, required=True)
    parser.add_argument("--image_min_value", help="minimum allowed pixel value", type=int, required=True)
    parser.add_argument("--image_size", help="image size", type=int, required=True)
    parser.add_argument("--batch_size", help="batch size", type=int, required=True)
    parser.add_argument("--num_classes", help="number of classes", type=int, required=True)
    parser.add_argument("--image_dir", help="path to synthetic image dataset ", type=str, required=True)
    parser.add_argument("--annotation_dir", help="path to conditioning annotation dataset", type=str, required=True)
    parser.add_argument("--sam_checkpoint", help="path to SAM model", type=str, required=True)
    parser.add_argument("--model_type", help="model type", type=str, required=True)
    parser.add_argument("--save_dir", help="path to save results", type=str, required=True)
    parser.add_argument("--gpu", help="gpu flag", action="store_true")

    args = parser.parse_args()
    check_args(args)

    dist.init_process_group(backend="nccl" if args.gpu else "gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_id = os.environ['TORCHELASTIC_RUN_ID']

    device = torch.device(f"cuda:{local_rank}" if args.gpu else "cpu")
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)

    data_loader = load_sam_dataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        batch_size=args.batch_size,
        image_max_value=args.image_max_value,
        image_min_value=args.image_min_value,
        image_size=args.image_size,
        num_classes=args.num_classes,
        shuffle=False,
        drop_last=False
    )

    batch_iou_scores = []
    for batch_idx, (images, annotations) in enumerate(data_loader):
        batched_input = convert_to_sam_batch(images, annotations, args.num_points, sam.image_encoder.img_size, device)
        batched_output = sam(batched_input, multimask_output=False)
        batched_pred_mask = []
        for output in batched_output:
            if output['masks'].shape[0] == 1:
                masks = output['masks']  # 1xCxHxW
            else:
                masks = output['masks'][1:, ...]  # BxCxHxW
            pred_mask = torch.zeros_like(masks[0]).type(torch.bool)  # CxHxW
            for mask in masks:
                pred_mask = pred_mask | mask  # CxHxW
            pred_mask = pred_mask.squeeze(0)  # HxW
            batched_pred_mask.append(pred_mask)
        batched_pred_mask = torch.stack(batched_pred_mask, dim=0)  # BxHxW
        batched_true_mask = torch.from_numpy(annotations[..., 1]).to(device=device).type(torch.bool)
        batch_iou_score = calculate_batch_iou_score(batched_pred_mask, batched_true_mask)
        batch_iou_scores.append(batch_iou_score)

        print(
            f"Rank: {rank}, "
            f"batch size: {batched_pred_mask.shape[0]}, "
            f"batch idx: {batch_idx + 1}/{len(data_loader)}, "
            f"IoU score: {batch_iou_score.item()}"
        )

    iou_score = torch.stack(batch_iou_scores, dim=0).mean()
    iou_score.contiguous()
    dist.all_reduce(iou_score, op=dist.ReduceOp.SUM)
    avg_iou_score = iou_score / world_size

    if rank == 0:
        print(f"Average IoU score: {avg_iou_score.item()}")
        file_path = Path(args.save_dir) / f"iou_score_{run_id}.txt"
        with open(file_path, "w") as f:
            f.write(f"{avg_iou_score.item()}")
        print(f"Saved IoU score to {file_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
