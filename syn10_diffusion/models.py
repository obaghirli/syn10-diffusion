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


class AttnBlock(nn.Module):
    def __init__(self, in_channels, head_channels):
        super().__init__()
        self.in_channels = in_channels
        self.head_channels = head_channels
        self.num_heads = in_channels // head_channels

        self.norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Conv1d(in_channels, 3 * in_channels, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        temp = x
        n, c, h, w = temp.shape
        assert c == self.in_channels
        temp = temp.reshape(n, c, h * w)  # (n, c, h*w)
        temp = self.norm(temp)  # (n, c, h*w)
        proj_in = self.proj_in(temp)  # (n, 3*c, h*w) (n, 3*head_channels*num_heads, h*w)
        qkv = proj_in.reshape(n * self.num_heads, 3 * self.head_channels, h * w)  # (n*num_heads, 3*head_channels, h*w)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # (n*num_heads, head_channels, h*w)
        scale = torch.sqrt(torch.sqrt(torch.tensor(1.) / self.head_channels))
        weights = torch.einsum('bct,bcs->bts', q * scale, k * scale)  # (n*num_heads, h*w, h*w)
        weights = weights.softmax(dim=-1)  # (n*num_heads, h*w, h*w)
        attns = torch.einsum('bts,bcs->bct', weights, v)  # (n*num_heads, head_channels, h*w)
        attns = attns.reshape(n, -1, h * w)  # (n, c, h*w)
        proj_out = self.proj_out(attns)  # (n, c, h*w)
        proj_out = proj_out.reshape(n, c, h, w)  # (n, c, h, w)
        return x + proj_out  # (n, c, h, w)


class UnetTest(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs['model_input_channels']
        self.out_channels = kwargs['model_output_channels']
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
        self.model_in_channels = kwargs['model_input_channels']
        self.model_out_channels = kwargs['model_output_channels']
        self.model_channels = kwargs['model_channels']
        self.model_resolution = kwargs['image_size']
        self.num_resnet_blocks = kwargs['num_resnet_blocks']
        self.channel_mult = kwargs['channel_mult']
        self.in_channel_mult = [1] + self.channel_mult
        self.num_resolutions = len(self.channel_mult)
        self.t_embed_mult = kwargs['t_embed_mult']
        self.y_embed_mult = kwargs['y_embed_mult']
        self.num_classes = kwargs['num_classes']
        self.attn_resolutions = kwargs['attn_resolutions']
        self.head_channels = kwargs['head_channels']
        self.dropout = kwargs['dropout']

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.model_in_channels, self.model_channels, kernel_size=3, padding=1)
        )

        time_embed_dim = int(self.model_channels * self.t_embed_mult)
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        resnet_in_channels, resnet_out_channels = (None, None)
        curr_resolution = self.model_resolution
        self.encoder = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            level_module = nn.Module()
            level_module.resnet_blocks = nn.ModuleList()
            level_module.attn_blocks = nn.ModuleList()
            level_module.downsample = None

            resnet_in_channels = self.model_channels * self.in_channel_mult[i_level]
            resnet_out_channels = self.model_channels * self.channel_mult[i_level]
            for i_resnet_block in range(self.num_resnet_blocks):
                level_module.resnet_blocks.append(
                    ResnetEncoderBlock(
                        in_channels=resnet_in_channels,
                        out_channels=resnet_out_channels,
                        emb_channels=time_embed_dim,
                        dropout=self.dropout
                    )
                )
                if curr_resolution in self.attn_resolutions:
                    level_module.attn_blocks.append(
                        AttnBlock(
                            in_channels=resnet_out_channels,
                            head_channels=self.head_channels
                        )
                    )
                resnet_in_channels = resnet_out_channels
            if i_level != self.num_resolutions - 1:
                level_module.downsample = Downsample(
                    channels=resnet_out_channels
                )
                curr_resolution //= 2
            self.encoder.append(level_module)

        assert resnet_in_channels is not None and resnet_out_channels is not None

        segmap_channels = 1 if self.num_classes == 2 else self.num_classes
        segmap_emb_channels = int(self.model_channels * self.y_embed_mult)

        self.middle = nn.ModuleList([
            ResnetDecoderBlock(
                in_channels=resnet_out_channels,
                out_channels=resnet_out_channels,
                segmap_channels=segmap_channels,
                segmap_emb_channels=segmap_emb_channels,
                t_emb_channels=time_embed_dim,
                dropout=self.dropout
            ),
            AttnBlock(
                in_channels=resnet_out_channels,
                head_channels=self.head_channels
            ),
            ResnetDecoderBlock(
                in_channels=resnet_out_channels,
                out_channels=resnet_out_channels,
                segmap_channels=segmap_channels,
                segmap_emb_channels=segmap_emb_channels,
                t_emb_channels=time_embed_dim,
                dropout=self.dropout
            )
        ])

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.model_channels),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, self.model_out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, y):
        emb = get_timestep_embeddings(t, self.model_channels)
        emb = self.time_embed(emb)

        hs = [self.in_conv(x)]



        out = self.out_layers(x)
        return out

