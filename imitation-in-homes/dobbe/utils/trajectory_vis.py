import collections

import einops
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_plots(ground_truths, predictions, sampled_images, to_plot=8, traj_index=0):
    """
    Generates plots comparing ground truth vs predictions and displays sampled images.

    Parameters:
    - ground_truths: A numpy array of shape (T, 7) containing ground truth actions.
    - predictions: A numpy array of shape (T, 7) with model's predictions.
    - sampled_images: A numpy array containing M sampled images.
    """

    # Initialize the figure
    fig = plt.figure(figsize=(15, 15))
    outer_grid = gridspec.GridSpec(8, 1, hspace=0.25, wspace=0.0)

    # Top grid for images
    M = sampled_images.shape[0]
    chosen_indices = np.linspace(0, M - 1, to_plot, dtype=int)
    chosen_images = sampled_images[chosen_indices]
    top_grid = gridspec.GridSpecFromSubplotSpec(
        1, len(chosen_images), subplot_spec=outer_grid[0, :], wspace=0.0
    )

    for i, img in enumerate(chosen_images):
        ax = plt.Subplot(fig, top_grid[i])
        ax.imshow(img)
        ax.axis("off")
        if i > 0:  # To further remove any potential whitespace
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        fig.add_subplot(ax)

    # For each action dimension, plot ground truth vs prediction
    dim = ground_truths.shape[1]
    for i in range(dim):
        ax = plt.Subplot(fig, outer_grid[i + 1, :])

        gt_values = ground_truths[:, i]
        pred_values = predictions[:, i]

        ax.plot(gt_values, "g", label="Ground Truth")
        ax.plot(pred_values, "r--", label="Prediction")

        ax.legend(loc="upper right")
        ax.set_title(f"Act Dim {i + 1}", fontsize=16)
        fig.add_subplot(ax)

    # save the plot
    fig.savefig(f"trajectory_{traj_index}.png")


@torch.no_grad()
def visualize_trajectory(
    model, test_dataset, device, buffer_size=6, n_visualized_trajectories=5
):
    """
    Visualizes the trajectory of the model on the test dataset.
    M images are sampled from the test dataset and displayed.
    T = end - start is the length of the trajectory.
    start and end (index) indicate the window of the trajectory to visualize.
    """
    action_preds = []
    ground_truth = []
    images = []
    image_buffer = collections.deque(maxlen=buffer_size)
    test_dataset.set_include_trajectory_end(True)

    print("Visualizing trajectories...")
    print("# frames in trajectory =", len(test_dataset))

    i = 0
    done_visualizing = 0
    while (done_visualizing < n_visualized_trajectories) and (i < len(test_dataset)):
        (input_images, terminate), *_, gt_actions = test_dataset[i]
        input_images = input_images.float() / 255.0
        image_buffer.append(input_images[-1])
        img = input_images[-1]
        images.append(einops.rearrange(img, "c h w -> h w c").cpu().detach().numpy())
        ground_truth.append(gt_actions[-1])
        model_input = (
            torch.stack(tuple(image_buffer), dim=0).unsqueeze(0).to(device),
            torch.tensor(gt_actions).unsqueeze(0).to(device),
        )

        breakpoint()
        out, _ = model.step(model_input)
        action_preds.append(out.squeeze().cpu().detach().numpy())

        if terminate:
            action_preds = np.array(action_preds)
            ground_truth = np.array(ground_truth)
            images = np.array(images)

            print(action_preds.shape, ground_truth.shape, images.shape)

            generate_plots(
                ground_truth, action_preds, images, traj_index=done_visualizing
            )
            done_visualizing += 1
            # Reset everything.
            action_preds = []
            ground_truth = []
            images = []
            image_buffer = collections.deque(maxlen=buffer_size)

        i += 1
