import os
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
from syn10_diffusion.logger import DistributedLogger, TBSummaryWriter
from syn10_diffusion import utils

utils.seed_all()


class Trainer:
    def __init__(self, model, diffusion, data, **params):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.run_id = params['run_id']
        self.run_dir = Path(params['artifact_dir']) / self.run_id
        self.num_epochs = params['num_epochs']
        self.grad_clip = params['grad_clip']
        self.p_uncond = params['p_uncond']
        self.checkpoint_freq = params['checkpoint_freq']
        self.tensorboard_freq = params['tensorboard_freq']
        self.step = 0
        self.start_epoch = 0

        self.dlogger = DistributedLogger()
        self.tb_writer = TBSummaryWriter(log_dir=self.run_dir)

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
        self.start_epoch = optimizer_checkpoint['epoch']

    def save_checkpoint(self, step, epoch):
        model_checkpoint_path = self.run_dir / f"model_checkpoint_{step}.pt.tar"
        optimizer_checkpoint_path = self.run_dir / f"optimizer_checkpoint_{step}.pt.tar"

        torch.save({
            'model_state_dict': self.ddp_model.module.state_dict(),
        }, model_checkpoint_path)

        torch.save({
            'step': step,
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, optimizer_checkpoint_path)

    def run(self):
        avg_loss = torch.zeros(1).to(self.local_rank)
        avg_mse_loss = torch.zeros(1).to(self.local_rank)
        avg_vlb_loss = torch.zeros(1).to(self.local_rank)

        for epoch in range(self.start_epoch, self.num_epochs):
            for i, (x, y) in enumerate(self.data):
                print(f"Rank: {self.rank}, batch size: {x.shape[0]}, epoch: {epoch}, step: {self.step}", end='\r')
                x, y = map(lambda tensor: tensor.to(self.local_rank), (x, y))
                n = x.shape[0]
                mask = torch.rand(size=(n, 1, 1, 1), device=y.device) >= self.p_uncond
                y = y * mask.float()

                t = torch.from_numpy(np.random.choice(len(self.diffusion.betas), size=(n,))).long().to(self.local_rank)

                loss_terms = self.diffusion.training_losses(self.ddp_model, x, t, model_kwargs={'y': y})

                loss = loss_terms['loss'].mean()
                mse = loss_terms['mse'].mean()
                vlb = loss_terms['vlb'].mean()

                avg_loss += loss.clone() / self.tensorboard_freq
                avg_mse_loss += mse.clone() / self.tensorboard_freq
                avg_vlb_loss += vlb.clone() / self.tensorboard_freq

                self.optimizer.zero_grad()
                loss.backward()

                try:
                    clip_grad_norm_(self.ddp_model.parameters(), self.grad_clip)
                except Exception:
                    pass

                self.optimizer.step()
                self.step += 1

                if self.step % self.checkpoint_freq == 0 and self.rank == 0:
                    self.save_checkpoint(self.step, epoch)
                if self.step % self.tensorboard_freq == 0 and self.rank == 0:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_mse_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_vlb_loss, op=dist.ReduceOp.SUM)
                    avg_loss /= self.world_size
                    avg_mse_loss /= self.world_size
                    avg_vlb_loss /= self.world_size
                    self.tb_writer.add_scalars(
                        "train", {
                            "avg_loss": avg_loss.item(),
                            "avg_mse_loss": avg_mse_loss.item(),
                            "avg_vlb_loss": avg_vlb_loss.item()
                        }, self.step
                    )
                    avg_loss.zero_()
                    avg_mse_loss.zero_()
                    avg_vlb_loss.zero_()
                dist.barrier()
