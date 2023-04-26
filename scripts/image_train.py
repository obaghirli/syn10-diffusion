import os
import sys
from pathlib import Path
import argparse

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from syn10_diffusion.diffusion import Diffusion
from syn10_diffusion.sat25k import load_sat25k
from syn10_diffusion.train_utils import Trainer
from syn10_diffusion.globals import Globals
from syn10_diffusion.logger import DistributedLogger
from syn10_diffusion import utils

utils.seed_all()


def resolve_params(parser_args, config):
    params = {}
    params.update(config)
    params.update(vars(parser_args))
    params.update({"run_id": parser_args.run_id or os.environ['TORCHELASTIC_RUN_ID']})
    params.update({"is_train": True})
    return params


def validate_args(parser_args):
    not_parser_run_id = parser_args.run_id is None
    not_parser_resume_step = parser_args.resume_step is None
    if not_parser_run_id ^ not_parser_resume_step:
        raise RuntimeError("Both run_id and resume_step must be specified to resume training")


def setup_directories(params):
    rank = int(os.environ['RANK'])
    if rank == 0:
        artifact_dir = Path(params['artifact_dir'])
        run_dir = artifact_dir / params['run_id']
        run_dir.mkdir(parents=True, exist_ok=True)


@record
def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="path to config file", type=str, required=True)
    parser.add_argument("--data_dir", help="path to data directory", type=str, required=True)
    parser.add_argument("--artifact_dir", help="path to output directory", type=str, required=True)
    parser.add_argument("--run_id", help="torch elastic run id (checkpoint id)", type=str)
    parser.add_argument("--resume_step", help="step to continue from", type=int)
    parser.add_argument("--test_model", help="test model to use, debug mode", type=str)

    parser_args = parser.parse_args()
    validate_args(parser_args)
    config = utils.parse_config(parser_args.config)
    params = resolve_params(parser_args, config)
    _globals = Globals()
    _globals.params = params
    setup_directories(params)

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])

    logger = DistributedLogger()
    logger.log_info(
        f"{'Resuming' if params['resume_step'] is not None else 'Starting'} job_id: {params['run_id']}"
        f"{' from step: ' + str(params['resume_step']) if params['resume_step'] is not None else ''}"
    )
    logger.log_info(str(params))
    logger.log_info("Loading data")
    data = load_sat25k(
        data_dir=params['data_dir'],
        batch_size=params['train_batch_size'],
        image_size=params['image_size'],
        image_channels=params['image_channels'],
        image_max_value=params['image_max_value'],
        image_min_value=params['image_min_value'],
        num_classes=params['num_classes'],
        is_train=params['is_train'],
        shuffle=True,
        drop_last=True,
    )

    logger.log_info("Creating diffusion")
    diffusion = Diffusion(**params)

    logger.log_info("Creating model")
    model_classes = utils.get_models()
    model_class = model_classes[params['test_model']] \
        if params['test_model'] is not None else model_classes["prod"]
    logger.log_info(f"Using model: {model_class.__name__}")
    model = model_class(**params).to(local_rank)
    model.train()

    logger.log_info("Creating trainer")
    trainer = Trainer(model=model, diffusion=diffusion, data=data, **params)

    logger.log_info("Starting training")
    trainer.run()

    logger.log_info("Training finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
