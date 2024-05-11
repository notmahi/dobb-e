import cv2
import numpy as np
from matplotlib import pyplot as plt


def cv2_imshow(im):
    disp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    dpi = 80
    height, width, depth = disp.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    plt.imshow(disp)
    plt.show()


def cv2_imcmp(im1, im2):
    disp1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    disp2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    dpi = 80
    height, width, depth = disp1.shape
    figsize = (8 / 3) * width / float(dpi), (8 / 3) * height / float(dpi)
    f, axarr = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0)
    axarr[0].set_xticklabels([])
    axarr[1].set_xticklabels([])
    axarr[0].imshow(disp1)
    axarr[1].imshow(disp2)
    plt.show()


def display_image_in_grid(grid, label_grid=None):
    x = len(grid)
    y = len(grid[0])

    fig, axarr = plt.subplots(x, y, figsize=(int(40 * (y / 6)), int(60 * (x / 11))))
    for idx1, row in enumerate(grid):
        if len(row[0].shape) == 3:
            axarr[idx1][0].imshow((row[0] * 255).astype(np.uint8), label="test")
            axarr[idx1][0].title.set_text("test, " + str(label_grid[idx1][0]))
            for idx2, img in enumerate(row[1:]):
                axarr[idx1][idx2 + 1].imshow((img * 255).astype(np.uint8), label="nbhr")
                axarr[idx1][idx2 + 1].title.set_text(
                    "nbhr, " + str(label_grid[idx1][idx2 + 1])
                )
        else:
            axarr[idx1][0].imshow(
                (row[0] * 255).astype(np.uint8), label="test", cmap="gray"
            )
            axarr[idx1][0].title.set_text("test")
            for idx2, img in enumerate(row[1:]):
                axarr[idx1][idx2 + 1].imshow(
                    (img * 255).astype(np.uint8), label="nbhr", cmap="gray"
                )
                axarr[idx1][idx2 + 1].title.set_text(
                    "nbhr, " + str(label_grid[idx1][idx2 + 1])
                )


def overlay_action(
    vector,  # action vector to be plotted
    img,  # image to be plotted on
    color=(255, 0, 0),
    plot_z=True,  # plot z component of action vector
    scale_percent=100,
    vector_scale=1,
    shift_start_point=False,
):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    vector = vector_scale * vector
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    start_point = (int(dim[0] / 2), int(height * 4 / 5))
    if shift_start_point:
        # shift start point to the right by 1/6 of the image width
        start_point = (int(dim[0] / 2 + dim[0] / 10), int(height * 4 / 5))

    scale = 100 * scale_percent / 100

    end_point = (
        int(start_point[0] + (vector[0]) * scale),
        int(start_point[1] - vector[1] * scale),
    )

    thickness = int(4 * scale_percent / 100)

    img = cv2.arrowedLine(img, start_point, end_point, color, thickness)

    if plot_z:
        start_point = (int(width * 4 / 5), int(height * 4 / 5))
        if shift_start_point:
            start_point = (int(width * 4 / 5 + dim[0] / 10), int(height * 4 / 5))

        scale = 100 * scale_percent / 100

        end_point = (
            int(start_point[0]),
            int(start_point[1] - vector[2] * scale * 2),
        )

        thickness = int(4 * scale_percent / 100)
        img = cv2.arrowedLine(img, start_point, end_point, color, thickness)

    return img
