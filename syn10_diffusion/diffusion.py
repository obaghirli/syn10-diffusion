import torch
import numpy as np
from typing import Tuple
from syn10_diffusion.utils import seed_all

seed_all()


def get_timestep_embeddings(timesteps: np.ndarray, out_dim: int, max_period: int = 10000) -> np.ndarray:
    """ Computes the timestep embeddings.

    Args:
        timesteps: A 1D array of timesteps.
        out_dim: The output dimension.
        max_period: The maximum period of the cosine function.

    Returns:  A 2D array of shape (len(timesteps), out_dim) containing the timestep embeddings.
    """
    assert len(timesteps.shape) == 1
    half_dim = out_dim // 2
    exponents = np.arange(half_dim) / (half_dim - 1)
    frequencies = np.exp(-np.log(max_period) * exponents)
    angles = timesteps[:, None] * frequencies[None]
    cos, sin = np.cos(angles), np.sin(angles)
    timestep_embeddings = np.concatenate((cos, sin), axis=1)
    assert timestep_embeddings.shape == (len(timesteps), out_dim)
    return timestep_embeddings


class Diffusion:
    def __init__(self, **kwargs):
        self.num_diffusion_timesteps = kwargs['num_diffusion_timesteps']
        self.max_beta = kwargs['max_beta']
        self.s = kwargs['s']
        self.lambda_variational = kwargs['lambda_variational']
        self.betas = self.get_beta_schedule(self.num_diffusion_timesteps, self.max_beta, self.s)
        self.alphas = 1 - self.betas
        self.sqrt_alphas = np.sqrt(self.alphas)
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(self.one_minus_alphas_cumprod)
        self.alphas_cumprod_prev = np.append(1, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.q_posterior_variance = self.one_minus_alphas_cumprod_prev / self.one_minus_alphas_cumprod * self.betas
        self.q_posterior_log_variance_clipped = np.log(np.append(self.q_posterior_variance[1], self.q_posterior_variance[1:]))
        self.q_posterior_mean_coef_x0 = self.sqrt_alphas_cumprod_prev / self.one_minus_alphas_cumprod * self.betas
        self.q_posterior_coef_xt = self.sqrt_alphas * self.one_minus_alphas_cumprod_prev / self.one_minus_alphas_cumprod
        self.recip_sqrt_alphas = 1.0 / self.sqrt_alphas
        self.recip_sqrt_alphas_cumprod = 1.0 / self.sqrt_alphas_cumprod
        self.recip_sqrt_one_minus_alphas_cumprod = 1.0 / self.sqrt_one_minus_alphas_cumprod
        self.log_betas = np.log(self.betas)

    def get_beta_schedule(self, num_diffusion_timesteps: int, max_beta: float = 0.999, s: float = 0.008) -> np.ndarray:
        """ Computes the beta schedule. Eq.17 from the Improved DDPM paper.

        Args:
            num_diffusion_timesteps: The number of diffusion timesteps.
            max_beta: The maximum beta value to prevent singularities at the end of the diffusion process near t=T.
            s: offset to prevent beta from being too small near t=0.

        Returns: A 1D array of shape (num_diffusion_timesteps,) containing the beta schedule.
        """
        t_prev = np.arange(num_diffusion_timesteps)
        alpha_bar_prev = np.cos((t_prev / num_diffusion_timesteps + s) / (1 + s) * np.pi / 2) ** 2

        t = t_prev + 1
        alpha_bar = np.cos((t / num_diffusion_timesteps + s) / (1 + s) * np.pi / 2) ** 2

        betas = 1 - alpha_bar / alpha_bar_prev
        betas = np.clip(betas, a_min=None, a_max=max_beta)

        assert np.all(betas > 0) and np.all(betas < 1)
        assert betas.shape == (num_diffusion_timesteps,)

        return betas

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """ Sample from q(x_t | x_0)

        Args:
            x_start: The starting point of the diffusion process. A tensor of shape (batch_size, num_channels, height, width).
            t: The timestep at which to sample. A 1D array of shape (batch_size,).
            noise: The noise to use for the sampling. A tensor of shape (batch_size, num_channels, height, width).

        Returns: A tensor of shape (batch_size, num_channels, height, width) containing the samples.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        if isinstance(noise, np.ndarray):
            noise = torch.from_numpy(noise)

        assert noise.shape == x_start.shape
        x_t = _slice(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
            _slice(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        assert x_t.shape == x_start.shape
        return x_t

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Compute the mean and variance of q(x_{t-1} | x_t, x_0)

        Args:
            x_start: The starting point of the diffusion process. A tensor of shape (batch_size, num_channels, height, width).
            x_t: The value of x at timestep t. A tensor of shape (batch_size, num_channels, height, width).
            t: The timestep at which to compute the posterior. A 1D array of shape (batch_size,).

        Returns:  A tuple of tensors containing the mean, variance, and log_variance_clipped of q(x_{t-1} | x_t, x_0).
        """
        assert x_start.shape == x_t.shape
        assert t.shape == (x_start.shape[0],)
        q_posterior_mean = _slice(self.q_posterior_mean_coef_x0, t, x_start.shape) * x_start + \
            _slice(self.q_posterior_coef_xt, t, x_t.shape) * x_t
        q_posterior_variance = _slice(self.q_posterior_variance, t, x_start.shape)
        q_posterior_log_variance_clipped = _slice(self.q_posterior_log_variance_clipped, t, x_start.shape)
        assert all(
            tensor.shape == x_start.shape for tensor in (
                q_posterior_mean,
                q_posterior_variance,
                q_posterior_log_variance_clipped
            )
        )
        return q_posterior_mean, q_posterior_variance, q_posterior_log_variance_clipped

    def predict_x_start_from_noise(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        assert t.shape == (x_t.shape[0],)
        x_start = _slice(self.recip_sqrt_alphas_cumprod, t, x_t.shape) * \
            (x_t - _slice(self.sqrt_one_minus_alphas_cumprod, t, eps.shape) * eps)
        assert x_start.shape == x_t.shape
        return x_start

    def p_mean_variance(self, x_t, t, eps, var_signal, clip_denoised=False):
        assert all(x_t.shape == tensor.shape for tensor in (eps, var_signal))
        x_start_pred = self.predict_x_start_from_noise(x_t, t, eps)
        if clip_denoised:
            x_start_pred = torch.clamp(x_start_pred, -1.0, 1.0)
        p_mean, _, _ = self.q_posterior_mean_variance(x_start_pred, x_t, t)

        v = (var_signal + 1.0) / 2.0
        min_log_variance = _slice(self.q_posterior_log_variance_clipped, t, v.shape)
        max_log_variance = _slice(self.log_betas, t, v.shape)
        p_log_variance = v * max_log_variance + (1.0 - v) * min_log_variance
        p_variance = torch.exp(p_log_variance)
        assert all(x_t.shape == tensor.shape for tensor in (p_mean, p_log_variance, p_variance))
        return p_mean, p_variance, p_log_variance

    def training_losses(self, model, x_start, t, noise=None, model_kwargs=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        if model_kwargs is None:
            model_kwargs = {}

        x_t = self.q_sample(x_start, t, noise)
        q_posterior_mean, _, q_posterior_log_variance_clipped = self.q_posterior_mean_variance(x_start, x_t, t)

        model_output = model(x_t, t.float(), model_kwargs['y'])
        eps, var_signal = torch.chunk(model_output, 2, dim=1)

        assert all(x_start.shape == tensor.shape for tensor in (eps, var_signal))
        assert var_signal.shape == q_posterior_log_variance_clipped.shape

        p_mean, _, p_log_variance = self.p_mean_variance(x_t, t, eps.detach(), var_signal, clip_denoised=True)

        # calculate mse loss
        mse = torch.mean((noise - eps)**2, dim=list(range(1, noise.ndim)))

        # calculate vlb loss
        q_posterior_dist = torch.distributions.normal.Normal(
            q_posterior_mean,
            torch.exp(0.5 * q_posterior_log_variance_clipped)
        )

        p_dist = torch.distributions.normal.Normal(
            p_mean,
            torch.exp(0.5 * p_log_variance)
        )

        kl = torch.mean(
            torch.distributions.kl.kl_divergence(
                q_posterior_dist,
                p_dist
            ), dim=list(range(1, x_start.ndim))
        )  # nats

        decoder_nll = torch.mean(
            -discretized_gaussian_log_likelihood(
                x_start,
                means=p_mean,
                log_scales=0.5 * p_log_variance
            ), dim=list(range(1, x_start.ndim))
        )  # nats

        vlb = torch.where((t == 0), decoder_nll, kl)

        total_loss = mse + self.lambda_variational * vlb
        terms = {
            'loss': total_loss,
            'mse': mse,
            'vlb': vlb,
            'kl': kl,
            'decoder_nll': decoder_nll,
            'var_signal': torch.mean(var_signal.detach(), dim=list(range(1, var_signal.ndim)))
        }
        return terms

    def p_sample(self, model, shape, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        assert isinstance(shape, (torch.Size, tuple, list))
        assert model.training is False

        device = next(model.parameters()).device
        y = model_kwargs['y']
        guidance = model_kwargs['guidance']
        g_ch = model_kwargs['model_output_channels'] // 2

        x = torch.randn(*shape).to(device)
        n = x.shape[0]

        sequence = range(len(self.betas))
        for timestep in reversed(list(sequence)):
            with torch.no_grad():
                t = torch.ones(n).to(device) * timestep
                model_output = model(x, t.float(), y)
                model_output_zero = model(x, t.float(), torch.zeros_like(y))
                model_output[:, :g_ch] = \
                    model_output_zero[:, :g_ch] + guidance * (model_output[:, :g_ch] - model_output_zero[:, :g_ch])
                eps, var_signal = torch.chunk(model_output, 2, dim=1)
                p_mean, _, p_log_variance = self.p_mean_variance(x, t.long(), eps, var_signal, clip_denoised=True)
                none_zero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
                x = p_mean + none_zero_mask * torch.exp(0.5 * p_log_variance) * torch.randn_like(x)
        return x

    def p_sample_trajectory(self, model, shape, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        assert isinstance(shape, (torch.Size, tuple, list))
        assert model.training is False

        device = next(model.parameters()).device
        y = model_kwargs['y']
        guidance = model_kwargs['guidance']
        g_ch = model_kwargs['model_output_channels'] // 2

        x = torch.randn(*shape).to(device)
        n = x.shape[0]

        save_trajectory = model_kwargs['save_trajectory']
        num_trajectory = len(save_trajectory)
        trajectory = torch.empty(
            (shape[0], shape[1]*num_trajectory, shape[2], shape[3]),
            dtype=torch.float32
        ).to(device)
        channel_offset = 0
        if 1000 in save_trajectory:
            print(f"time step: {1000}")
            print(f"register time step: {1000}")
            trajectory[:, channel_offset:shape[1], :, :] = x
            channel_offset += shape[1]
        sequence = range(len(self.betas))
        for timestep in reversed(list(sequence)):
            print(f"time step: {timestep}")
            with torch.no_grad():
                t = torch.ones(n).to(device) * timestep
                model_output = model(x, t.float(), y)
                model_output_zero = model(x, t.float(), torch.zeros_like(y))
                model_output[:, :g_ch] = \
                    model_output_zero[:, :g_ch] + guidance * (model_output[:, :g_ch] - model_output_zero[:, :g_ch])
                eps, var_signal = torch.chunk(model_output, 2, dim=1)
                p_mean, _, p_log_variance = self.p_mean_variance(x, t.long(), eps, var_signal, clip_denoised=True)
                none_zero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
                x = p_mean + none_zero_mask * torch.exp(0.5 * p_log_variance) * torch.randn_like(x)
                if timestep in save_trajectory:
                    print(f"register time step: {timestep}")
                    trajectory[:, channel_offset:channel_offset+shape[1], :, :] = x
                    channel_offset += shape[1]
        return trajectory

    def p_sample_interpolate(self, model, shape, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        assert isinstance(shape, (torch.Size, tuple, list))
        assert model.training is False

        device = next(model.parameters()).device
        x = model_kwargs['x']  # on model device
        y = model_kwargs['y']  # on model device
        num_steps = model_kwargs['num_steps']
        guidance = model_kwargs['guidance']
        g_ch = model_kwargs['model_output_channels'] // 2
        n = x.shape[0]

        sequence = range(len(self.betas))
        for timestep in reversed(list(sequence)):
            if timestep > num_steps:
                continue
            with torch.no_grad():
                t = torch.ones(n).to(device) * timestep
                model_output = model(x, t.float(), y)
                model_output_zero = model(x, t.float(), torch.zeros_like(y))
                model_output[:, :g_ch] = \
                    model_output_zero[:, :g_ch] + guidance * (model_output[:, :g_ch] - model_output_zero[:, :g_ch])
                eps, var_signal = torch.chunk(model_output, 2, dim=1)
                p_mean, _, p_log_variance = self.p_mean_variance(x, t.long(), eps, var_signal, clip_denoised=True)
                none_zero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
                x = p_mean + none_zero_mask * torch.exp(0.5 * p_log_variance) * torch.randn_like(x)
        return x


def _slice(arr: np.ndarray, timesteps: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    sliced_arr = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while sliced_arr.ndim < len(broadcast_shape):
        sliced_arr = sliced_arr[..., np.newaxis]
    return sliced_arr.expand(broadcast_shape)


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
