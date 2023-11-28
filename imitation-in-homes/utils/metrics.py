# calculate mean and std of dataset actions

import numpy as np
import torch
import tqdm


def get_act_mean_std(dataset):
    act_dim = dataset[0][1].shape[1]
    act_mean = torch.zeros(act_dim)
    act_std = torch.zeros(act_dim)
    count = 0
    for dataset in dataset.datasets:
        for label in dataset.labels:
            label = torch.from_numpy(label)
            act_mean += label
            act_std += label**2
            count += 1

    act_mean /= count
    act_std /= count
    act_std = torch.sqrt(act_std - act_mean**2)
    return act_mean, act_std


def get_act_statistics_from_loader(loader):
    action_sum = None
    action_count = 0
    action_max = None
    action_min = None
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader))):
        action = batch[1]
        if action_sum is None:
            action_sum = action.sum(dim=0).sum(dim=0)
            action_max = action.max(dim=0).values.max(dim=0).values
            action_min = action.min(dim=0).values.min(dim=0).values
        else:
            action_sum += action.sum(dim=0).sum(dim=0)
            action_max = torch.max(
                action_max, action.max(dim=0).values.max(dim=0).values
            )
            action_min = torch.min(
                action_min, action.min(dim=0).values.min(dim=0).values
            )
        action_count += action.shape[0] * action.shape[1]

    action_mean = action_sum / action_count
    action_std_sum = None
    action_count
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader))):
        action = batch[1]
        if action_std_sum is None:
            action_std_sum = ((action - action_mean) ** 2).sum(dim=0).sum(dim=0)
        else:
            action_std_sum += ((action - action_mean) ** 2).sum(dim=0).sum(dim=0)
    action_std = (action_std_sum / action_count) ** 0.5

    print("Mean: ", action_mean)
    print("Std: ", action_std)
    print("Max: ", action_max)
    print("Min: ", action_min)

    return action_mean, action_std, action_max, action_min
