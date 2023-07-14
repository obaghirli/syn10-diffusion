import os
import sys
import time
import json
from pathlib import Path
from itertools import islice
import argparse

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import numpy as np

from syn10_diffusion.diffusion import Diffusion
from syn10_diffusion.sat25k import load_sat25k
from syn10_diffusion import models
from syn10_diffusion.globals import Globals
from syn10_diffusion import utils

utils.seed_all()


def resolve_params(parser_args, config):
    params = {}
    params.update(config)
    params.update(vars(parser_args))
    params.update({"run_id": os.environ['TORCHELASTIC_RUN_ID']})
    params.update({"is_train": False})
    return params


def validate_args(parser_args):
    if not Path(parser_args.config).exists():
        raise RuntimeError(f"Config file {parser_args.config} not found")
    if not Path(parser_args.model_path).exists():
        raise RuntimeError(f"Model file {parser_args.model_path} not found")
    if not Path(parser_args.data_dir).exists():
        raise RuntimeError(f"Data directory {parser_args.data_dir} not found")
    if parser_args.test_model is not None:
        if parser_args.test_model not in models.get_models().keys():
            raise ValueError(f"Unknown test model {parser_args.test_model}. "
                             f"Available test models: {list(set(models.get_models().keys()) - set('prod'))}")
    for timestep in parser_args.save_trajectory:
        if timestep < 0 or timestep > 1000:
            raise ValueError(f"Invalid timestep {timestep}. Must be in range [0, 1000]")
    if parser_args.save_trajectory != sorted(parser_args.save_trajectory, reverse=True):
        raise ValueError("save_trajectory must be sorted in reverse order")


def setup_directories(params):
    rank = int(os.environ['RANK'])
    if rank == 0:
        artifact_dir = Path(params['artifact_dir'])
        artifact_dir.mkdir(parents=True, exist_ok=True)


@record
def main():
    parser = argparse.ArgumentParser(description="Sampling")
    parser.add_argument("--config", help="path to config file", type=str, required=True)
    parser.add_argument("--model_path", help="path to model file", type=str, required=True)
    parser.add_argument("--data_dir", help="path to data directory", type=str, required=True)
    parser.add_argument("--artifact_dir", help="path to output directory", type=str, required=True)
    parser.add_argument("--save_trajectory", help="save intermediate results", type=int, nargs="+", required=True)
    parser.add_argument("--num_batches", help="number of batches", type=int, default=1)
    parser.add_argument("--test_model", help="test model to use, debug mode", type=str)

    parser_args = parser.parse_args()
    validate_args(parser_args)
    config = utils.parse_config(parser_args.config)
    utils.validate_config(config)
    params = resolve_params(parser_args, config)
    _globals = Globals()
    _globals.params = params
    setup_directories(params)

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    assert world_size == 1, "Only single GPU supported"
    num_batches = params.get("num_batches")
    assert num_batches == 1, "Only single batch supported"

    print(f"Starting job id: {params['run_id']}")
    print(str(params))
    print("Loading data")
    data_loader = load_sat25k(
        data_dir=params['data_dir'],
        batch_size=params['sample_batch_size'],
        image_size=params['image_size'],
        image_channels=params['image_channels'],
        image_max_value=params['image_max_value'],
        image_min_value=params['image_min_value'],
        num_classes=params['num_classes'],
        is_train=params['is_train'],
        shuffle=True,
        drop_last=False,
    )

    if num_batches is not None:
        data_loader = islice(data_loader, num_batches)

    print("Creating sampler")
    diffusion = Diffusion(**params)

    print("Loading model")
    model_classes = models.get_models()
    model_class = model_classes[params["test_model"]] \
        if params["test_model"] is not None else model_classes["prod"]
    print(f"Using model: {model_class.__name__}")
    model = model_class(**params).to(local_rank)
    model.eval()
    model_checkpoint = torch.load(Path(params['model_path']), map_location=f"cuda:{local_rank}")
    model.load_state_dict(model_checkpoint['model_state_dict'])
    print(f"Loaded model from: {Path(params['model_path']).resolve()}")
    print("Starting sampling")
    start_time = time.time()

    all_samples = []
    for i, (x, y) in enumerate(data_loader):
        print(
            f"Rank: {rank}, "
            f"batch size: {x.shape[0]}, "
            f"i_batch: {i + 1}/{len(data_loader) if num_batches is None else num_batches}", end='\r'
        )
        y = y.to(local_rank)
        sample = diffusion.p_sample_trajectory(
            model,
            x.shape,
            model_kwargs={
                'y': y,
                'guidance': params['guidance'],
                'model_output_channels': params['model_output_channels'],
                'save_trajectory': params['save_trajectory']
            }
        )
        sample = (sample + 1.0) / 2.0 * (params['image_max_value'] - params['image_min_value']) \
            + params['image_min_value']
        sample = sample.clamp(params['image_min_value'], params['image_max_value'])
        sample = sample.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(world_size)]
        dist.all_gather(gathered_samples, sample)
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

    end_time = time.time()
    if rank == 0:
        save_path = Path(params['artifact_dir']) / f"trajectory"
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving trajectories...")
        arr = np.concatenate(all_samples, axis=0).astype(np.uint16)
        filename = f"trajectory_{params['run_id']}"
        np.save(str(save_path / f"{filename}.npy"), arr)
        log_file = save_path / f"trajectory_{params['run_id']}.json"
        with open(log_file, "w") as f:
            json.dump(params, f, indent=4)
        print("Sampling finished")
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")
        print(f"Saved trajectory images to: {str(save_path.resolve())}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    sys.exit(main())
