import os
import sys
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record


import yaml

from unet import UnetModel
from diffusion import Diffusion
from datasets import load_sat25k
from train_utils import Trainer

from typing import Optional


def parse_config(config_path: Optional[str]):
    if config_path is None:
        return {}
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def resolve_params(parser_args, config, dist_args):
    params = {}
    params.update(config)
    params.update(vars(parser_args))
    params.update(dist_args)
    if parser_args.run_id is not None:
        params['run_id'] = parser_args.run_id
    validate_params(params, parser_args, config, dist_args)
    return params


def validate_params(params, parser_args, config, dist_args):
    if params['artifact_dir'] is None:
        raise RuntimeError("Artifact directory must be specified")
    not_parser_run_id = parser_args.run_id is None
    not_parser_resume_step = parser_args.resume_step is None
    if not_parser_run_id ^ not_parser_resume_step:
        raise RuntimeError("Both run_id and resume_step must be specified")
    assert isinstance(params['grad_clip'], float), "grad_clip must be a float"


def setup_directories(params):
    artifact_dir = Path(params['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_dir = artifact_dir / params['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)


@record
def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="path to config file", type=str)
    parser.add_argument("--run_id", help="torch elastic run id (checkpoint id)", type=str)
    parser.add_argument("--resume_step", help="step to continue from", type=int)

    parser_args = parser.parse_args()
    config = parse_config(parser_args.config)

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    run_id = os.environ['TORCHELASTIC_RUN_ID']

    dist_args = {
        "local_rank": local_rank,
        "run_id": run_id
    }

    params = resolve_params(parser_args, config, dist_args)
    setup_directories(params)

    train_loader = load_sat25k(
        data_dir=params['data_dir'],
        batch_size=params['train_batch_size'],
        image_size=params['image_size'],
        image_channels=params['image_channels'],
        image_max_value=params['image_max_value'],
        image_min_value=params['image_min_value'],
        num_classes=params['num_classes'],
        is_train=True,
        shuffle=True
    )

    diffusion = Diffusion(**params)
    model = UnetModel(**params).to(local_rank)
    trainer = Trainer(model=model, diffusion=diffusion, data=train_loader, **params)
    trainer.run()


if __name__ == "__main__":
    print(f"Starting job: {os.environ['TORCHELASTIC_RUN_ID']}")
    sys.exit(main())
