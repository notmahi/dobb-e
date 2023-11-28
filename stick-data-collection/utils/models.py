import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms as T


def create_fc(
    input_dim, output_dim, hidden_dims, use_batchnorm, dropout=None, is_moco=False
):
    if hidden_dims is None:
        return nn.Sequential(*[nn.Linear(input_dim, output_dim)])

    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    if dropout is not None:
        layers.append(nn.Dropout(p=dropout))

    for idx in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
        layers.append(nn.ReLU())

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[idx + 1]))

        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))

    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if is_moco:
        layers.append(nn.BatchNorm1d(hidden_dims[-1], affine=False))
    return nn.Sequential(*layers)


def compute_std_loss(rep, epsilon=1e-04):
    rep = rep - rep.mean(dim=0)
    rep_std = torch.sqrt(rep.var(dim=0) + epsilon)
    return torch.mean(F.relu(1 - rep_std)) / 2.0


def off_diagonal(rep_cov):
    n, _ = rep_cov.shape
    return rep_cov.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_cov_loss(rep, feature_size):
    rep_cov = (rep.T @ rep) / (rep.shape[0] - 1)
    return off_diagonal(rep_cov).pow_(2).sum().div(feature_size)


def vicreg_loss(input_rep, output_rep, feature_size, sim_coef, std_coef, cov_coef):
    sim_loss = F.mse_loss(input_rep, output_rep)
    std_loss = compute_std_loss(input_rep) + compute_std_loss(output_rep)
    cov_loss = compute_cov_loss(input_rep, feature_size) + compute_cov_loss(
        output_rep, feature_size
    )

    final_loss = (sim_coef * sim_loss) + (std_coef * std_loss) + (cov_coef * cov_loss)
    loss_dict = {
        "train_loss": final_loss.item(),
        "sim_loss": sim_loss.item(),
        "std_loss": std_loss.item(),
        "cov_loss": cov_loss.item(),
    }

    return final_loss, loss_dict


class BCrep(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(BCrep, self).__init__()
        self.f1 = nn.Linear(input_dim, 1024)
        self.f2 = nn.Linear(1024, 512)
        self.f3 = nn.Linear(512, 128)
        self.f4 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        x = F.relu(x)
        x = self.f4(x)
        return x


class BClayer(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(BClayer, self).__init__()
        self.f1 = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = self.f1(x)
        return x


class VICReg(nn.Module):
    def __init__(
        self, backbone, projector, augment_fn, sim_coef=25, std_coef=25, cov_coef=1
    ):
        super(VICReg, self).__init__()

        # Networks
        self.backbone = backbone
        self.projector = projector
        self.augment_fn = augment_fn

        # Loss parameters
        self.sim_coef = sim_coef
        self.std_coef = std_coef
        self.cov_coef = cov_coef

    def get_image_representation(self, image):
        augmented_image = self.augment_fn(image)
        representation = self.projector(self.backbone(augmented_image))
        return representation

    def forward(self, image):
        first_projection = self.get_image_representation(image)
        second_projection = self.get_image_representation(image)

        loss, loss_info = vicreg_loss(
            input_rep=first_projection,
            output_rep=second_projection,
            feature_size=first_projection.shape[-1],
            sim_coef=self.sim_coef,
            std_coef=self.std_coef,
            cov_coef=self.cov_coef,
        )

        return loss, loss_info

    def get_encoder_weights(self):
        return self.backbone.state_dict()


class Identity(nn.Module):
    """
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GripperNet(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super(GripperNet, self).__init__()

        self.resnet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self.resnet18.fc = Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet18(x)
        x = torch.sigmoid(self.regressor(x))
        return x


class BcModel(nn.Module):
    def __init__(self, input_dim):
        super(BcModel, self).__init__()
        self.f1 = nn.Linear(input_dim, input_dim)
        self.f2 = nn.Linear(input_dim, 1024)
        self.f3 = nn.Linear(1024, 256)
        self.f4 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        x = F.relu(x)
        x = self.f4(x)
        return x
