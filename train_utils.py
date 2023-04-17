import os
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path


class Trainer:
    def __init__(self, model, diffusion, data, **params):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.run_id = params['run_id']
        self.run_dir = Path(params['artifact_dir']) / self.run_id
        self.step = 0
        self.grad_clip = params['grad_clip']
        self.checkpoint_freq = params['checkpoint_freq']

        if params['resume_step'] is not None:
            self.load_model_checkpoint(params['resume_step'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])

        if params['resume_step'] is not None:
            self.load_optimizer_checkpoint(params['resume_step'])

        self.ddp_model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank
        )

    def load_model_checkpoint(self, resume_step):
        model_checkpoint_path = self.run_dir / f"model_checkpoint_{resume_step}.pt.tar"
        if not model_checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint {model_checkpoint_path} does not exist")
        model_checkpoint = torch.load(model_checkpoint_path, map_location=f"cuda:{self.local_rank}")
        self.model.load_state_dict(model_checkpoint['model_state_dict'])

    def load_optimizer_checkpoint(self, resume_step):
        optimizer_checkpoint_path = self.run_dir / f"optimizer_checkpoint_{resume_step}.pt.tar"
        if not optimizer_checkpoint_path.exists():
            raise FileNotFoundError(f"Optimizer checkpoint {optimizer_checkpoint_path} does not exist")
        optimizer_checkpoint = torch.load(optimizer_checkpoint_path, map_location=f"cuda:{self.local_rank}")
        self.optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])
        self.step = optimizer_checkpoint['step']

    def save_checkpoint(self, step):
        if self.rank == 0:
            model_checkpoint_path = self.run_dir / f"model_checkpoint_{step}.pt.tar"
            optimizer_checkpoint_path = self.run_dir / f"optimizer_checkpoint_{step}.pt.tar"

            torch.save({
                'model_state_dict': self.ddp_model.module.state_dict(),
            }, model_checkpoint_path)

            torch.save({
                'step': step,
                'optimizer_state_dict': self.optimizer.state_dict()
            }, optimizer_checkpoint_path)

        dist.barrier()

    def run(self):
        while True:
            x, y = map(lambda tensor: tensor.to(self.local_rank), next(self.data))
            n = x.shape[0]
            t = torch.from_numpy(np.random.choice(len(self.diffusion.betas), size=(n,))).long().to(self.local_rank)

            loss_terms = self.diffusion.training_losses(self.ddp_model, x, t, model_kwargs={'y': y})
            loss = loss_terms['loss'].mean()

            self.optimizer.zero_grad()
            loss.backward()

            try:
                clip_grad_norm_(self.ddp_model.parameters(), self.grad_clip)
            except Exception:
                pass

            self.optimizer.step()
            self.step += 1
            print(f"step: {self.step}", end='\r')

            if self.step % self.checkpoint_freq == 0:
                self.save_checkpoint(self.step)

