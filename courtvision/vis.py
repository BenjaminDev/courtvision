import os
from pathlib import Path
from typing import Union

import cv2
import kornia as K
import matplotlib.pyplot as plt
import numpy as np

# import rerun_sdk
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")


def plot_coords(img: np.array, src_coords: np.array, show: bool = True, thickness=2):
    """Draws lines on 'img'

    Args:
        img (np.array): _description_
        src_coords (np.array): _description_
        show (bool, optional): _description_. Defaults to True.
    """
    src_coords = src_coords.astype(int)
    cv2.polylines(img, [src_coords], True, (0, 255, 0), thickness=2)
    if show:
        from matplotlib import pyplot as plt

        plt.imshow(img)


def plot_3d_points(
    x,
    y,
    z,
    plt_axis: None | Axes3D = None,
    colors: None | list = None,
    view_init: tuple[float, float, float] = (90.0, 0.0, 0.0),
) -> Axes3D:
    fig = None
    if plt_axis is None:
        fig = plt.figure()
        plt_axis = fig.add_subplot(111, projection="3d")
    # plot the points
    if colors is not None and len(colors) == len(x):
        plt_axis.scatter(x, y, z, c=colors)
    else:
        plt_axis.scatter(x, y, z, c="y", s=10)
    # set the axis labels
    plt_axis.set_xlabel("X")
    plt_axis.set_ylabel("Y")
    plt_axis.set_zlabel("Z")
    plt_axis.set_aspect("equal")
    plt_axis.view_init(*view_init)
    return plt_axis, fig


def plot_3d_lines(
    xs: np.array,
    ys: np.array,
    zs: np.array,
    plt_axis: None | Axes3D = None,
    view_init: tuple[float, float, float] = (90.0, 0.0, 0.0),
) -> Axes3D:
    """plots lines on a Axes3D

    Args:
        xs (np.array): array of x start and stop coordinates
        ys (np.array): array of y start and stop coordinates
        zs (np.array): array of z start and stop coordinates
        plt_axis (None | Axes3D, optional): Axes3D to draw on. If not given one will be created Defaults to None.
        view_init (tuple[float, float, float], optional): Position of the camera to view. Defaults to (90., 0., 0.).

    Returns:
        Axes3D: matplotlib ax that can be added to or shown.
    """

    fig = None
    if plt_axis is None:
        fig = plt.figure()
        plt_axis = fig.add_subplot(111, projection="3d")
    for i in range(len(xs)):
        plt_axis.plot(xs[i], ys[i], zs[i], c="b")
    # set the axis labels
    plt_axis.set_xlabel("X")
    plt_axis.set_ylabel("Y")
    plt_axis.set_zlabel("Z")
    plt_axis.set_aspect("equal")
    plt_axis.view_init(*view_init)
    return plt_axis, fig


def load_timg(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"  # nosec
    # load image with OpenCV
    if isinstance(file_name, Path):
        file_name = file_name.as_posix()
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # convert image to torch tensor
    tensor = K.image_to_tensor(img, None).float() / 255.0
    return K.color.bgr_to_rgb(tensor)


def points_to_heat_map(
    named_points: dict,
    image: None | np.ndarray = None,
    height: None | int | float = None,
    width: None | int | float = None,
    padding: int = 10,
    offset: int = 5,
    normalised: bool = False,
) -> np.array:
    if isinstance(height, float):
        height = int(height)
    if isinstance(width, float):
        width = int(width)
    if image is None:
        image = np.zeros(shape=(height, width), dtype=np.float32)
    if normalised:
        points = (
            np.array([(v[0] * width, v[1] * height) for _, v in named_points.items()])
            + offset
        )
    else:
        points = (
            np.array([v for _, v in named_points.items()], dtype=np.float32) + offset
        )

    # Calculate the 2D histogram of the points
    heatmap, x_edges, y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=[width, height],
        range=[[0, width + padding], [0, height + padding]],
    )

    # Smooth the heatmap using a Gaussian filter
    heatmap = np.float32(ndimage.gaussian_filter(heatmap, sigma=10, radius=300))
    return heatmap.T, points


def draw_points(
    image, points, color=(0, 255, 0), radius=10, thickness=-1, fontScale=1, labels=None
):
    for i, p1 in enumerate(points[:]):
        cv2.circle(
            img=image,
            center=[int(o) for o in p1],
            radius=radius,
            color=color,
            thickness=thickness,
        )
        if labels is not None:
            cv2.putText(
                image,
                labels[i],
                (int(p1[0]), int(p1[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (0, 0, 0),
                2,
            )
    return image


def plot_n_images_in_a_grid(images: list[np.array], n_cols: int = 3):
    """draws a grid of images

    Args:
        images (list[np.array]): images to draw on - this is inplace
        n_cols (int, optional): number of cols. Defaults to 3.

    Returns:
        tuple(fig, ax): matplotlib fig and ax
    """
    n_cols = min(n_cols, len(images))
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    if n_rows == 1:
        ax = ax.reshape(1, n_cols)
    for i, img in enumerate(images):
        ax[i // n_cols, i % n_cols].imshow(img)
    return fig, ax


# from courtvision.data import KeypointValue, RectValue


def draw_rect(image: Union[np.ndarray, torch.tensor], bboxes: list["RectValue"]):

    from kornia.utils import draw_rectangle, image_to_tensor

    if isinstance(image, np.ndarray):
        image = image_to_tensor(image)

    # rect = torch.stack([torch.tensor(
    #         [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height]
    #     ).unsqueeze(0) for bbox in bboxes])
    return draw_rectangle(image, bboxes, fill=True, color=torch.tensor([0.0, 1.0, 0.0]))
