import os
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from syn10_diffusion.logger import DistributedLogger
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
        self.eval_freq = params['eval_freq']
        self.guidance = params['guidance']
        self.model_output_channels = params['model_output_channels']
        self.image_max_value = params['image_max_value']
        self.image_min_value = params['image_min_value']
        self.step = 0
        self.local_step = 0
        self.start_epoch = 0
        self.dlogger = DistributedLogger()

        self.ddp_model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank
        )

        self.optimizer = torch.optim.AdamW(
            self.ddp_model.module.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=params['lr_scheduler_t_0'],
            T_mult=params['lr_scheduler_t_mult']
        )

        self.ema = EMA(
            self.ddp_model,
            decay=params['ema_decay'],
            delay=params['ema_delay']
        )

        if params['resume_step'] is not None:
            self.load_model_checkpoint(params['resume_step'])
            self.load_optimizer_checkpoint(params['resume_step'])
            self.load_lr_scheduler_checkpoint(params['resume_step'])
            if self.step >= self.ema.delay:
                self.load_ema_checkpoint(params['resume_step'])

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

    @torch.no_grad()
    def compute_norms(self):
        param_norm = 0.0
        grad_norm = 0.0
        for param in self.ddp_model.module.parameters():
            if param.requires_grad:
                param_norm += torch.linalg.vector_norm(param).item() ** 2
                if param.grad is not None:
                    grad_norm += torch.linalg.vector_norm(param.grad).item() ** 2
        return np.sqrt(param_norm), np.sqrt(grad_norm)

    @torch.no_grad()
    def evaluate(self, x, y):
        self.ddp_model.eval()
        sample = self.diffusion.p_sample(
            self.ddp_model.module,
            x.shape,
            model_kwargs={
                'y': y,
                'guidance': self.guidance,
                'model_output_channels': self.model_output_channels
            }
        )
        sample = (sample + 1.0) / 2.0
        sample = sample.clamp(0.0, 1.0)
        self.ddp_model.train()
        return sample

    def run(self):
        avg_loss = torch.FloatTensor([0.]).to(self.local_rank)
        avg_mse_loss = torch.FloatTensor([0.]).to(self.local_rank)
        avg_vlb_loss = torch.FloatTensor([0.]).to(self.local_rank)
        avg_kl_loss = torch.FloatTensor([0.]).to(self.local_rank)
        avg_decoder_nll_loss = torch.FloatTensor([0.]).to(self.local_rank)
        avg_var_signal = torch.FloatTensor([0.]).to(self.local_rank)
        t_counter = torch.zeros((len(self.diffusion.betas),), dtype=torch.long).to(self.local_rank)

        for epoch in range(self.start_epoch, self.num_epochs):
            for i, (x, y) in enumerate(self.data, self.local_step):
                if i == self.iters:
                    break

                print(
                    f"Rank: {self.rank}, "
                    f"batch size: {x.shape[0]}, "
                    f"i_batch: {i + 1}/{self.iters}, "
                    f"epoch: {epoch}, "
                    f"step: {self.step}, "
                    f"local_step: {i}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']}"
                )

                if self.step == self.ema.delay:
                    self.ema.build_shadow()

                x, y = map(lambda tensor: tensor.to(self.local_rank), (x, y))
                n = x.shape[0]
                mask = torch.rand(size=(n, 1, 1, 1), device=y.device) >= self.p_uncond
                y = y * mask.float()

                t = torch.from_numpy(np.random.choice(len(self.diffusion.betas), size=(n,))).long().to(self.local_rank)
                t_counter[t] += 1

                loss_terms = self.diffusion.training_losses(self.ddp_model, x, t, model_kwargs={'y': y})

                loss = loss_terms['loss'].mean()
                mse = loss_terms['mse'].mean()
                vlb = loss_terms['vlb'].mean()
                kl = loss_terms['kl'].mean()
                decoder_nll = loss_terms['decoder_nll'].mean()
                var_signal = loss_terms['var_signal'].mean()

                print(
                    f"Rank: {self.rank}, "
                    f"loss: {loss}, "
                    f"mse: {mse}, "
                    f"vlb: {vlb}, "
                    f"kl: {kl}, "
                    f"decoder_nll: {decoder_nll}, "
                    f"var_signal: {var_signal}"
                )

                avg_loss += loss.detach().clone() / self.tensorboard_freq
                avg_mse_loss += mse.detach().clone() / self.tensorboard_freq
                avg_vlb_loss += vlb.detach().clone() / self.tensorboard_freq
                avg_kl_loss += kl.detach().clone() / self.tensorboard_freq
                avg_decoder_nll_loss += decoder_nll.detach().clone() / self.tensorboard_freq
                avg_var_signal += var_signal.detach().clone() / self.tensorboard_freq

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
                    dist.all_reduce(avg_kl_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_decoder_nll_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_var_signal, op=dist.ReduceOp.SUM)
                    t_counter_clone = t_counter.clone()
                    dist.all_reduce(t_counter_clone, op=dist.ReduceOp.SUM)
                    avg_loss /= self.world_size
                    avg_mse_loss /= self.world_size
                    avg_vlb_loss /= self.world_size
                    avg_kl_loss /= self.world_size
                    avg_decoder_nll_loss /= self.world_size
                    avg_var_signal /= self.world_size

                    param_norm, grad_norm = self.compute_norms()

                    with SummaryWriter(log_dir=self.run_dir) as tb_writer:
                        tb_writer.add_scalar("avg_loss", avg_loss.item(), self.step)
                        tb_writer.add_scalar("avg_mse_loss", avg_mse_loss.item(), self.step)
                        tb_writer.add_scalar("avg_vlb_loss", avg_vlb_loss.item(), self.step)
                        tb_writer.add_scalar("avg_kl_loss", avg_kl_loss.item(), self.step)
                        tb_writer.add_scalar("avg_decoder_nll_loss", avg_decoder_nll_loss.item(), self.step)
                        tb_writer.add_scalar("avg_var_signal", avg_var_signal.item(), self.step)
                        tb_writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], self.step)
                        tb_writer.add_scalar("param_norm", param_norm, self.step)
                        tb_writer.add_scalar("grad_norm", grad_norm, self.step)
                        tb_writer.add_figure("t_counter", t_counter_fig(t_counter_clone), self.step)

                        with torch.no_grad():
                            for name, param in self.ddp_model.module.named_parameters():
                                if param.requires_grad:
                                    tb_writer.add_histogram(name, param.detach().cpu().numpy(), self.step)

                    avg_loss.zero_()
                    avg_mse_loss.zero_()
                    avg_vlb_loss.zero_()
                    avg_kl_loss.zero_()
                    avg_decoder_nll_loss.zero_()
                    avg_var_signal.zero_()

                if self.step % self.eval_freq == 0 and self.rank == 0 and self.step > 0:
                    print(f"Rank: {self.rank}, Evaluating...")
                    sample = self.evaluate(x, y)
                    if sample.shape[1] >= 3:
                        sample = torch.narrow(sample, 1, 0, 3)
                    else:
                        sample = torch.narrow(sample, 1, 0, 1)
                    with SummaryWriter(log_dir=self.run_dir) as tb_writer:
                        tb_writer.add_images("sample", sample, self.step)

                dist.barrier()
                self.step += 1
            self.local_step = 0


def t_counter_fig(t_counter):
    fig, ax = plt.subplots()
    x = t_counter.cpu().numpy()
    bins = np.arange(np.floor(np.min(x)) - 0.5, np.ceil(np.max(x)) + 1.5)
    counts, limits = np.histogram(x, bins=bins)
    ax.bar(limits[:-1], counts, align='edge')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of timesteps')
    return fig
