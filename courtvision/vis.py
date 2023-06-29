import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import cv2
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch
import ultralytics
from kornia.geometry import unproject_points
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

from courtvision.data import PadelCourt
from courtvision.geometry import (
    camera_space_to_world_space,
    compute_ray_intersecting_plane,
)

colours_per_player_idx = defaultdict(lambda: (255, 255, 255))
colours_per_player_idx.update(
    {
        0: (0, 255, 0),
        1: (0, 0, 255),
        2: (255, 0, 0),
        3: (255, 255, 0),
        4: (255, 0, 255),
        5: (0, 255, 255),
    }
)


def log_court_layout(
    camera_matrix: np.array,
    image_width: int,
    image_height: int,
    translation_vector: np.array,
    rotation_vector: np.array,
    court_mesh_path: Path,
):
    """_summary_

    Args:
        camera_matrix (np.array): _description_
        image_width (int): _description_
        image_height (int): _description_
        translation_vector (np.array): _description_
        rotation_vector (np.array): _description_
        court_mesh_path (Path): _description_
    """
    rr.log_pinhole(
        "world/camera/image",
        child_from_parent=camera_matrix,
        width=image_width,
        height=image_height,
        timeless=True,
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    rr.log_transform3d(
        "world/camera",
        transform=rr.TranslationAndMat3(
            translation=translation_vector.squeeze(),
            matrix=rotation_matrix,
        ),
        from_parent=True,
    )
    rr.log_point("world/front_left", np.array([0, 0, 0]))
    # TODO: this should be refectored to use the court_size
    rr.log_point("world/front_right", np.array([100, 0, 0]))
    rr.log_mesh_file(
        "world",
        mesh_format=rr.MeshFormat("GLB"),
        mesh_path=court_mesh_path,
    )


def log_player_detections(
    detections: list[ultralytics.yolo.engine.results.Results],
    camera_matrix: np.array,
    translation_vector: np.array,
    rotation_vector: np.array,
    clip_uid: str,
):
    player_maker_radius = 5.0
    if len(detections) > 1:
        raise NotImplementedError("Only batches of size 1 are supported for now")
    result = detections[0]

    for det in result.boxes.data:
        # TODO: Fix this hack. Use?
        # x1, y1, x2, y2, idx, *_ = det
        if len(det) == 7:
            x1, y1, x2, y2, idx, conf, cls = det
        else:
            x1, y1, x2, y2, idx, conf = det
            cls = 0
        rr.log_rect(
            f"world/camera/image/player_{int(idx)}",
            (x1, y1, (x2 - x1), (y2 - y1)),
            color=colours_per_player_idx[int(idx)],
        )
        # Compute the 3D point of the player's feet
        # Use 2 depth values to unproject the point from the image plane to the camera plane
        depths = torch.tensor(
            [[1.0 * PadelCourt.court_scale, 20.0 * PadelCourt.court_scale]]
        ).T  # Depth values in [mm * PadelCourt.court_scale]
        mid_feets = torch.tensor([((x1 + x2) / 2, (y2 + y2) / 2)]).repeat(
            depths.shape[0], 1
        )
        mid_feets_base_camera_space = unproject_points(
            point_2d=mid_feets, camera_matrix=camera_matrix, depth=depths
        ).squeeze(0)
        # Using the Translation and Rotation Vector of the camera, transform the point from camera space to world space
        mid_feet_base_world_space = camera_space_to_world_space(
            mid_feets_base_camera_space.squeeze(0).numpy().T,
            translation_vector,
            rotation_vector,
        )
        # Compute the intersection of the ray formed by the camera position and the 3D point with the plane
        intersection = compute_ray_intersecting_plane(
            point_a_on_ray=mid_feet_base_world_space[0].reshape(3, 1),
            point_b_on_ray=mid_feet_base_world_space[1].reshape(3, 1),
        )

        rr.log_point(
            f"world/player_{int(idx)}",
            intersection,
            radius=player_maker_radius,
            color=colours_per_player_idx[int(idx)],
        )


def log_ball_detections(
    detections: list[dict[str, torch.Tensor]],
    clip_uid: str,
):
    boxes = detections[0]["boxes"]
    scores = detections[0]["scores"]
    labels = detections[0]["labels"]
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        rr.log_rect(
            "world/camera/image/ball_detections",
            rect=(
                box[0].item(),
                box[1].item(),
                box[2].item() - box[0].item(),
                box[3].item() - box[1].item(),
            ),
            timeless=False,
        )


def plot_coords(img: np.array, src_coords: np.array, show: bool = True, thickness=2):
    """Draws lines on 'img'.

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
    plt_axis.margins(x=0)
    plt_axis.view_init(*view_init)
    return plt_axis, fig


def load_timg(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor."""
    img = load_image(file_name)
    # convert image to torch tensor
    return K.image_to_tensor(img, None).float() / 255.0


def load_image(file_name):
    """Loads the image with OpenCV."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"  # nosec
    # load image with OpenCV
    if isinstance(file_name, Path):
        file_name = file_name.as_posix()
    return cv2.cvtColor(cv2.imread(file_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


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
