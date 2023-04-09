import os
import sys
from pathlib import Path
import argparse

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import yaml
import numpy as np

from unet import UnetModel
from diffusion import Diffusion


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="path to config file", type=str)
    parser.add_argument("--use_ddp", help="whether to use ddp", action="store_true")
    args = parser.parse_args()

    config = None
    if args.config is not None:
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file {args.config} not found")
        config = parse_config(Path(args.config))

    params = {}
    if config is not None:
        params.update(config)
    params.update(vars(args))

    # load data
    train_loader = ...

    # create diffusion
    diffusion = Diffusion(**params)

    # create model
    model = UnetModel(**params)
    if all([torch.cuda.is_available(), args.use_ddp]):
        model = DDP(model)

    # create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=params['lr'])

    global_step = 0
    while True:
        x, y = next(train_loader)
        n = x.shape[0]
        t = torch.from_numpy(np.random.choice(len(diffusion.num_diffusion_timesteps), size=(n,))).long()

        loss_terms = diffusion.training_losses(model, x, t, model_kwargs={'y': y})
        loss = loss_terms['loss']

        optim.zero_grad()
        loss.backward()

        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        except Exception:
            pass

        optim.step()
        global_step += 1


if __name__ == "__main__":
    sys.exit(main())
