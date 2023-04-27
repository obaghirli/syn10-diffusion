import torch
from torch.nn import functional as F
import torch.nn as nn
from syn10_diffusion.utils import seed_all

seed_all()


def get_timestep_embeddings(timesteps, embedding_dim, max_period=10_000):
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period)) * torch.arange(half_dim) / (half_dim - 1))
    angles = timesteps[..., None].float() * freqs[None, ...]
    timestep_embeddings = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    assert timestep_embeddings.shape == (timesteps.shape[0], embedding_dim)
    return timestep_embeddings


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResnetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * out_channels),
        )

        self.out_norm = nn.GroupNorm(32, out_channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        h = x
        h = self.in_layers(h)
        proj = self.embed_layers(emb)
        while len(proj.shape) < len(h.shape):
            proj = proj[..., None]
        scale, shift = torch.chunk(proj, 2, dim=1)
        h = self.out_norm(h) * (1.0 + scale) + shift
        h = self.out_rest(h)
        return h + self.skip_connection(x)


class SPADE(nn.Module):
    def __init__(self, in_channels, segmap_channels, segmap_emb_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(segmap_channels, segmap_emb_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.conv_gamma = nn.Conv2d(segmap_emb_channels, in_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(segmap_emb_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        x = self.norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='nearest')
        segmap = self.shared(segmap)
        gamma = self.conv_gamma(segmap)
        beta = self.conv_beta(segmap)
        return x * (1.0 + gamma) + beta


class ResnetDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            segmap_channels,
            segmap_emb_channels,
            t_emb_channels,
            dropout
    ):
        super().__init__()
        self.in_layes = nn.Sequential(
            SPADE(in_channels, segmap_channels, segmap_emb_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_channels, 2 * out_channels)
        )

        self.out_norm = SPADE(out_channels, segmap_channels, segmap_emb_channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb, segmap):
        h = x
        h = self.in_layers(h, segmap)
        proj = self.embed_layers(emb)
        while len(proj.shape) < len(h.shape):
            proj = proj[..., None]
        scale, shift = torch.chunk(proj, 2, dim=1)
        h = self.out_norm(h, segmap) * (1.0 + scale) + shift
        h = self.out_rest(h)
        return h + self.skip_connection(x)


class UnetTest(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs['model_in_ch']
        self.out_channels = kwargs['model_out_ch']
        self.nn.Conv2d = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, t, y):
        x = self.nn.Conv2d(x)
        return x


class UnetProd(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs['model_in_ch']
        self.out_channels = kwargs['model_out_ch']
        self.nn.Conv2d = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, t, y):
        x = self.nn.Conv2d(x)
        return x

