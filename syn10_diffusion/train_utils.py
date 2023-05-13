import os
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
from syn10_diffusion.logger import DistributedLogger, TBSummaryWriter
from syn10_diffusion import utils
from syn10_diffusion.ema import EMA

utils.seed_all()


class Trainer:
    def __init__(self, model, diffusion, data, **params):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.iters = len(data)
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
        self.local_step = 0
        self.start_epoch = 0

        self.dlogger = DistributedLogger()
        self.tb_writer = TBSummaryWriter(log_dir=self.run_dir)

        self.ddp_model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        if params['resume_step'] is not None:
            self.load_model_checkpoint(params['resume_step'])

        self.optimizer = torch.optim.AdamW(
            self.ddp_model.module.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        if params['resume_step'] is not None:
            self.load_optimizer_checkpoint(params['resume_step'])

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=params['lr_scheduler_t_0'],
            T_mult=params['lr_scheduler_t_mult']
        )
        if params['resume_step'] is not None:
            self.load_lr_scheduler_checkpoint(params['resume_step'])

        self.ema = EMA(self.ddp_model, decay=params['ema_decay'], delay=params['ema_delay'])
        if params['resume_step'] is not None and self.step >= self.ema.delay:
            self.load_ema_checkpoint(params['resume_step'])

        if params['resume_step'] is not None:
            self.local_step += 1
            if self.local_step >= self.iters:
                self.local_step = 0
                self.start_epoch += 1
            self.step += 1

    def load_model_checkpoint(self, resume_step):
        model_checkpoint_path = self.run_dir / f"model_checkpoint_{resume_step}.pt.tar"
        if not model_checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint {model_checkpoint_path} does not exist")
        model_checkpoint = torch.load(model_checkpoint_path, map_location=f"cuda:{self.local_rank}")
        self.ddp_model.module.load_state_dict(model_checkpoint['model_state_dict'])

    def load_optimizer_checkpoint(self, resume_step):
        optimizer_checkpoint_path = self.run_dir / f"optimizer_checkpoint_{resume_step}.pt.tar"
        if not optimizer_checkpoint_path.exists():
            raise FileNotFoundError(f"Optimizer checkpoint {optimizer_checkpoint_path} does not exist")
        optimizer_checkpoint = torch.load(optimizer_checkpoint_path, map_location=f"cuda:{self.local_rank}")
        self.optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])

    def load_lr_scheduler_checkpoint(self, resume_step):
        lr_scheduler_checkpoint_path = self.run_dir / f"lr_scheduler_checkpoint_{resume_step}.pt.tar"
        if not lr_scheduler_checkpoint_path.exists():
            raise FileNotFoundError(f"LR scheduler checkpoint {lr_scheduler_checkpoint_path} does not exist")
        lr_scheduler_checkpoint = torch.load(lr_scheduler_checkpoint_path, map_location=f"cuda:{self.local_rank}")
        self.lr_scheduler.load_state_dict(lr_scheduler_checkpoint['lr_scheduler_state_dict'])
        self.local_step = lr_scheduler_checkpoint['local_step']
        self.step = lr_scheduler_checkpoint['step']
        self.start_epoch = lr_scheduler_checkpoint['epoch']

    def load_ema_checkpoint(self, resume_step):
        ema_chckpoint_path = self.run_dir / f"ema_checkpoint_{resume_step}.pt.tar"
        if not ema_chckpoint_path.exists():
            raise FileNotFoundError(f"EMA checkpoint {ema_chckpoint_path} does not exist")
        ema_checkpoint = torch.load(ema_chckpoint_path, map_location=f"cuda:{self.local_rank}")
        self.ema.load_state_dict(ema_checkpoint['model_state_dict'])

    def save_checkpoint(self, local_step, step, epoch):
        model_checkpoint_path = self.run_dir / f"model_checkpoint_{step}.pt.tar"
        optimizer_checkpoint_path = self.run_dir / f"optimizer_checkpoint_{step}.pt.tar"
        lr_scheduler_checkpoint_path = self.run_dir / f"lr_scheduler_checkpoint_{step}.pt.tar"
        ema_checkpoint_path = self.run_dir / f"ema_checkpoint_{step}.pt.tar"

        torch.save({
            'model_state_dict': self.ddp_model.module.state_dict(),
        }, model_checkpoint_path)

        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict()
        }, optimizer_checkpoint_path)

        torch.save({
            'local_step': local_step,
            'step': step,
            'epoch': epoch,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }, lr_scheduler_checkpoint_path)

        if step >= self.ema.delay:
            torch.save({
                'model_state_dict': self.ema.state_dict()
            }, ema_checkpoint_path)

    def run(self):
        avg_loss = torch.FloatTensor([0.]).to(self.local_rank)
        avg_mse_loss = torch.FloatTensor([0.]).to(self.local_rank)
        avg_vlb_loss = torch.FloatTensor([0.]).to(self.local_rank)

        for epoch in range(self.start_epoch, self.num_epochs):
            for i, (x, y) in enumerate(self.data, self.local_step):
                if i == self.iters:
                    break

                print(f"Rank: {self.rank}, "
                      f"batch size: {x.shape[0]}, "
                      f"i_batch: {i + 1}/{self.iters}, "
                      f"epoch: {epoch}, "
                      f"local_step: {i}, "
                      f"step: {self.step}")

                if self.step == self.ema.delay:
                    self.ema.build_shadow()

                x, y = map(lambda tensor: tensor.to(self.local_rank), (x, y))
                n = x.shape[0]
                mask = torch.rand(size=(n, 1, 1, 1), device=y.device) >= self.p_uncond
                y = y * mask.float()

                t = torch.from_numpy(np.random.choice(len(self.diffusion.betas), size=(n,))).long().to(self.local_rank)

                loss_terms = self.diffusion.training_losses(self.ddp_model, x, t, model_kwargs={'y': y})

                loss = loss_terms['loss'].mean()
                mse = loss_terms['mse'].mean()
                vlb = loss_terms['vlb'].mean()

                avg_loss += loss.detach().clone() / self.tensorboard_freq
                avg_mse_loss += mse.detach().clone() / self.tensorboard_freq
                avg_vlb_loss += vlb.detach().clone() / self.tensorboard_freq

                self.optimizer.zero_grad()
                loss.backward()

                try:
                    clip_grad_norm_(self.ddp_model.module.parameters(), self.grad_clip)
                except Exception:
                    pass

                self.optimizer.step()
                self.lr_scheduler.step(epoch + i / self.iters)
                self.ema.step()

                if self.step % self.checkpoint_freq == 0 and self.rank == 0 and self.step > 0:
                    self.save_checkpoint(i, self.step, epoch)
                if self.step % self.tensorboard_freq == 0 and self.rank == 0 and self.step > 0:
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
                            "avg_vlb_loss": avg_vlb_loss.item(),
                            "lr": self.optimizer.param_groups[0]['lr']
                        }, self.step
                    )
                    avg_loss.zero_()
                    avg_mse_loss.zero_()
                    avg_vlb_loss.zero_()
                dist.barrier()
                self.step += 1
            self.local_step = 0
