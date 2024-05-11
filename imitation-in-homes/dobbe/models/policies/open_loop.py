import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm


class VINNOpenLoop(nn.Module):
    def __init__(self, encoder, k=5, enc_weight_pth=None, use_vinn=False, cfg=None):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.use_vinn = use_vinn

        self.k = k

        if enc_weight_pth is not None:
            self.encoder.load_state_dict(
                torch.load(enc_weight_pth, map_location="cpu")["model"]
            )

        self.representations = None
        self.actions = None
        self.imgs = None
        softmax = nn.Softmax(dim=1)
        self.dist_scale_func = lambda x: (softmax(-x))
        self.encoder.eval()
        self.device = "cpu"
        self.encoder.to(self.device)
        self.img_transform = T.Resize((256, 256), antialias=True)

        self.open_loop = False
        self.idx = 0

    def to(self, device):
        self.device = device
        self.encoder.to(device)

        return super().to(device)

    def set_dataset(self, dataloader):
        self.train_dataset = dataloader.dataset
        if self.use_vinn:
            for i, (image, label) in tqdm(enumerate(dataloader)):
                image = image.float() / 255.0
                image = image.to(self.device)
                label = torch.Tensor(label).to("cpu").detach().squeeze()

                x = (image, label)
                representation = self.encoder(x).to("cpu").detach().squeeze(dim=1)
                if self.representations is None:
                    self.representations = representation
                    self.actions = label
                    image = image.to("cpu").detach().numpy()
                    self.imgs = list(image)
                else:
                    self.representations = torch.cat(
                        (self.representations, representation), 0
                    )
                    self.actions = torch.cat((self.actions, label), 0)
                    image = image.to("cpu").detach().numpy()
                    self.imgs.extend(list(image))

    def step(self, img, **kwargs):
        logs = {}
        if self.use_vinn:
            normalized_image = self.img_transform(img[0].squeeze(0))
            if not self.open_loop:
                self.encoder.eval()
                with torch.no_grad():
                    act, indices = self(img, return_indices=True)
                    act = act.squeeze().detach()
                    act[:-1] = 0
                    act[-1] = 1

                    self.neighbor_1_idx = indices[0][0]

                action_tensor = torch.zeros(7)
                action_tensor[-1] = 1
                self.open_loop = True
                return action_tensor, logs, indices
            else:
                _, action = self.train_dataset[self.neighbor_1_idx + self.idx]
                action_tensor = torch.tensor(action).squeeze()
                self.idx += 1
                return action_tensor, logs, None
        else:
            _, action = self.train_dataset[self.idx]
            action_tensor = torch.tensor(action).squeeze()
            self.idx += 1
            return action_tensor, logs, None

    def __call__(self, batch_images, k=None, return_indices=False):
        if k is None:
            k = self.k

        all_distances = torch.zeros(
            (batch_images[0].shape[0], self.representations.shape[0])
        )

        batch_rep = self.encoder(batch_images).squeeze(dim=1).detach().to(self.device)
        dat_rep = self.representations.to(self.device)
        all_distances = torch.cdist(batch_rep, dat_rep).to("cpu")

        top_k_distances, indices = torch.topk(all_distances, k, dim=1, largest=False)
        top_k_actions = self.actions[indices].to(self.device)

        weights = self.dist_scale_func(top_k_distances).to(self.device)

        pred = torch.sum(
            top_k_actions * weights.unsqueeze(-1), dim=1
        )  # weighted average

        if return_indices:
            return pred, indices

        return pred

    def reset(self):
        self.open_loop = False
        self.idx = 0
