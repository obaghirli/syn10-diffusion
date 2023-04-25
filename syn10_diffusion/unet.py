import torch
from syn10_diffusion.utils import seed_all

seed_all()


class UnetModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs['model_in_ch']
        self.out_channels = kwargs['model_out_ch']
        self.conv2d = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, t, y):
        x = self.conv2d(x)
        return x

