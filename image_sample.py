import os
import sys
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import numpy as np
import yaml
from PIL import Image
from unet import UnetModel
from diffusion import Diffusion
from datasets import load_sat25k
from globals import Globals
from logger import DistributedLogger
from typing import Optional, Union

from utils import seed_all

seed_all()


def parse_config(config_path: Optional[str]):
    if config_path is None:
        return {}
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def resolve_params(parser_args, config):
    params = {}
    params.update(config)
    params.update(vars(parser_args))
    params.update({"run_id": os.environ['TORCHELASTIC_RUN_ID']})
    params.update({"is_train": False})
    return params


def validate_args(parser_args):
    if not Path(parser_args.model_path).exists():
        raise FileNotFoundError(f"Model file {parser_args.model_path} not found")


def setup_directories(params):
    rank = int(os.environ['RANK'])
    if rank == 0:
        artifact_dir = Path(params['artifact_dir'])
        run_dir = artifact_dir / params['run_id']
        samples_dir = run_dir / "images"
        labels_dir = run_dir / "annotations"
        samples_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)


@record
def main():
    parser = argparse.ArgumentParser(description="Sampling")
    parser.add_argument("--config", help="path to config file", type=str, required=True)
    parser.add_argument("--model_path", help="path to model file", type=str, required=True)
    parser.add_argument("--data_dir", help="path to data directory", type=str, required=True)
    parser.add_argument("--artifact_dir", help="path to output directory", type=str, required=True)

    parser_args = parser.parse_args()
    validate_args(parser_args)
    config = parse_config(parser_args.config)
    params = resolve_params(parser_args, config)
    _globals = Globals()
    _globals.params = params
    setup_directories(params)

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    logger = DistributedLogger()
    logger.log_info(f"Starting job id: {params['run_id']}")
    logger.log_info(str(params))
    logger.log_info("Loading data")
    data = load_sat25k(
        data_dir=params['data_dir'],
        batch_size=params['sample_batch_size'],
        image_size=params['image_size'],
        image_channels=params['image_channels'],
        image_max_value=params['image_max_value'],
        image_min_value=params['image_min_value'],
        num_classes=params['num_classes'],
        is_train=params['is_train'],
        shuffle=False,
        drop_last=True,
    )

    logger.log_info("Creating sampler")
    diffusion = Diffusion(**params)

    logger.log_info("Loading model")
    model = UnetModel(**params).to(local_rank)
    model.eval()
    model_checkpoint = torch.load(Path(params['model_path']), map_location=f"cuda:{local_rank}")
    model.load_state_dict(model_checkpoint['model_state_dict'])

    logger.log_info("Starting sampling")
    all_samples = []
    all_labels = []

    for i, (x, y) in enumerate(data):
        y = y.to(local_rank)
        sample = diffusion.p_sample(
            model,
            x.shape,
            model_kwargs={
                'y': y,
                'guidance': params['guidance'],
                'model_out_ch': params['model_out_ch']
            }
        )
        sample = (sample + 1.0) / 2.0 * (params['image_max_value'] - params['image_min_value']) \
            + params['image_min_value']
        sample = sample.clamp(params['image_min_value'], params['image_max_value']).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        y = y.to(torch.uint8)
        if params['num_classes'] > 2:
            classes = torch.arange(params['num_classes'], device=local_rank)
            classes = classes.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            classes = classes.expand(y.shape)
            y = (classes * y).sum(dim=1, keepdim=True)
        y = y.permute(0, 2, 3, 1)
        y = y.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(world_size)]
        dist.all_gather(gathered_samples, sample)
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = [torch.zeros_like(y) for _ in range(world_size)]
        dist.all_gather(gathered_labels, y)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

    if rank == 0:
        logger.log_info("Saving samples")
        arr = np.concatenate(all_samples, axis=0)
        label_arr = np.concatenate(all_labels, axis=0)
        for i in range(arr.shape[0]):
            sample = Image.fromarray(arr[i])
            if params['num_classes'] == 2:
                label = Image.fromarray(255 * label_arr[i].squeeze(), mode='L')
            else:
                label = Image.fromarray(label_arr[i].squeeze(), mode='L')
            sample.save(Path(params['artifact_dir']) / params['run_id'] / "images" / f"sample_{i}.png")
            label.save(Path(params['artifact_dir']) / params['run_id'] / "annotations" / f"label_{i}.png")

    dist.barrier()

    logger.log_info("Sampling finished")
    dist.destroy_process_group()


if __name__ == '__main__':
    sys.exit(main())
