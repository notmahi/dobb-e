import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class DepthBC(nn.Module):
    def __init__(self, output_dim=256, kernel_size=(12, 16), patch_dim=-1):
        nn.Module.__init__(self)
        self.kernel_size = kernel_size

        self.output_dim = output_dim
        self.patch_dim = patch_dim

        self.transform = T.Compose(
            [T.Resize((256, 256), antialias=True), T.CenterCrop((224, 224))]
        )
        self.avgpool = nn.AvgPool2d(4, stride=4)
        self.crop = T.CenterCrop((16, 16))

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        b, t, _, h, _ = x.shape
        nan_mask = torch.isnan(x)
        x[nan_mask] = 0.0
        x = einops.rearrange(x, "b t c h w -> (b t) c h w")

        if self.patch_dim == -1:
            x = self.transform(x)
            x = self.avgpool(x)
            x = self.crop(x)
            x = x.flatten(2).median(dim=2).values.squeeze(-1)
            x = einops.rearrange(x, "(b t) -> b t", b=b, t=t)
            x = einops.repeat(x, "b t -> b t d", d=self.output_dim)
        else:
            x = self.avgpool(x)
            x = self.unfold(x).median(dim=1).values
            x = einops.rearrange(x, "(b t) e -> b t e", b=b, t=t)
            x = x.repeat(1, 1, self.output_dim // self.patch_dim**2)

        x = x.to(torch.float32)
        return x
