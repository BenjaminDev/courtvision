import cv2
import numpy as np


def plot_coords(img: np.array, src_coords: np.array, show: bool = True):
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


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_points(
    x,
    y,
    z,
    plt_axis: None | Axes3D = None,
    view_init: tuple[float, float, float] = (90.0, 0.0, 0.0),
) -> Axes3D:
    if plt_axis is None:
        fig = plt.figure()
        plt_axis = fig.add_subplot(111, projection="3d")
    # plot the points
    plt_axis.scatter(x, y, z, c="y")
    # set the axis labels
    plt_axis.set_xlabel("X")
    plt_axis.set_ylabel("Y")
    plt_axis.set_zlabel("Z")
    plt_axis.set_aspect("equal")
    plt_axis.view_init(*view_init)
    return plt_axis


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
    return plt_axis
