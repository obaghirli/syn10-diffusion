import torch


class UnetModel(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, t, y=None):
        x = self.conv2d(x)
        return x

