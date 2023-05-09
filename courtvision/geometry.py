from dataclasses import dataclass

import numpy as np

court_scale = 100


@dataclass
class PadelCourt:
    # REF: https://www.lta.org.uk/4ad2a4/siteassets/play/padel/file/lta-padel-court-guidance.pdf
    width: float = 10.0 * court_scale
    length: float = 20.0 * court_scale
    backwall_height: float = 3.0 * court_scale
    serve_line_from_back_line: float = 2.0 * court_scale
    line_width: float = 0.05

    @classmethod
    @property
    def center_line(cls) -> np.array:
        return np.array(
            [(cls.width / 2, cls.length), (cls.width / 2, 0)], dtype=np.int32
        ).reshape(-1, 1, 2)

    @classmethod
    @property
    def net_line(cls) -> np.array:
        return np.array(
            [(0, cls.length / 2), (cls.width, cls.length / 2)], dtype=np.int64
        ).reshape(-1, 1, 2)

    @classmethod
    @property
    def near_serve_line(cls):
        return np.array(
            [
                (0, cls.length - cls.serve_line_from_back_line),
                (cls.width, cls.length - cls.serve_line_from_back_line),
            ],
            np.int32,
        ).reshape(-1, 1, 2)

    @classmethod
    @property
    def far_serve_line(cls):
        return np.array(
            [
                (0, cls.serve_line_from_back_line),
                (cls.width, cls.serve_line_from_back_line),
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

    @classmethod
    @property
    def front_left(cls):
        return (0.0, 0.0)

    @classmethod
    @property
    def front_right(cls):
        return (cls.width, 0)

    @classmethod
    @property
    def top_front_left_vertical_plane(cls):
        # x, z
        return (0.0, cls.backwall_height)

    @classmethod
    @property
    def top_front_right_vertical_plane(cls):
        # x, z
        return (cls.width, cls.backwall_height)

    @classmethod
    @property
    def back_left(cls):
        return (0.0, cls.length)

    @classmethod
    @property
    def back_right(cls):
        return (cls.width, cls.length)

    @classmethod
    @property
    def left_near_serve_line(cls):
        return (0.0, cls.serve_line_from_back_line)

    @classmethod
    @property
    def right_near_serve_line(cls):
        return (cls.width, cls.serve_line_from_back_line)

    @classmethod
    @property
    def left_far_serve_line(cls):
        return (0.0, cls.length - cls.serve_line_from_back_line)

    @classmethod
    @property
    def right_far_serve_line(cls):
        return (cls.width, cls.length - cls.serve_line_from_back_line)

    @classmethod
    @property
    def m_top_front_left():
        # TODO: add thes
        raise NotImplementedError()

    @classmethod
    @property
    def n_top_front_right():
        raise NotImplementedError()

    @classmethod
    @property
    def o_top_back_left():
        raise NotImplementedError()

    @classmethod
    @property
    def p_top_back_right():
        raise NotImplementedError()

    @classmethod
    @property
    def q_top_net_line_left():
        raise NotImplementedError()

    @classmethod
    @property
    def r_top_net_line_right():
        raise NotImplementedError()

    # Normalised:
    @classmethod
    @property
    def center_line_n(cls) -> np.array:
        return np.array(
            [
                ((cls.width / 2) / cls.width, cls.length / cls.length),
                ((cls.width / 2) / cls.width, 0),
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

    @classmethod
    @property
    def net_line_n(cls) -> np.array:
        return np.array(
            [
                (0, (cls.length / 2) / cls.length),
                (cls.width / cls.width, (cls.length / 2) / cls.length),
            ],
            dtype=np.int64,
        ).reshape(-1, 1, 2)

    @classmethod
    @property
    def front_left_n(cls):
        return (cls.front_left[0] / cls.width, cls.front_left[1] / cls.length)

    @classmethod
    @property
    def front_right_n(cls):
        return (cls.front_right[0] / cls.width, cls.front_right[1] / cls.length)

    @classmethod
    @property
    def top_front_left_vertical_plane_n(cls):
        # x, z
        return (0.0, 0.0)

    @classmethod
    @property
    def top_front_right_vertical_plane_n(cls):
        # x, z
        return (cls.width / cls.width, 0.0)

    @classmethod
    @property
    def front_left_vertical_plane_n(cls):
        # x, z
        return (0.0, cls.backwall_height / cls.backwall_height)

    @classmethod
    @property
    def front_right_vertical_plane_n(cls):
        # x, z
        return (cls.width / cls.width, cls.backwall_height / cls.backwall_height)

    @classmethod
    @property
    def back_left_n(cls):
        return (cls.back_left[0] / cls.width, cls.back_left[1] / cls.length)

    @classmethod
    @property
    def back_right_n(cls):
        return (cls.back_right[0] / cls.width, cls.back_right[1] / cls.length)

    @classmethod
    @property
    def left_near_serve_line_n(cls):
        return (
            cls.left_near_serve_line[0] / cls.width,
            cls.left_near_serve_line[1] / cls.length,
        )

    @classmethod
    @property
    def right_near_serve_line_n(cls):
        return (
            cls.right_near_serve_line[0] / cls.width,
            cls.right_near_serve_line[1] / cls.length,
        )

    @classmethod
    @property
    def left_far_serve_line_n(cls):
        return (
            cls.left_far_serve_line[0] / cls.width,
            cls.left_far_serve_line[1] / cls.length,
        )

    @classmethod
    @property
    def right_far_serve_line_n(cls):
        return (
            cls.right_far_serve_line[0] / cls.width,
            cls.right_far_serve_line[1] / cls.length,
        )

    # a_front_left
    # b_front_right
    # c_back_left
    # d_back_right
    # e_left_near_serve_line
    # f_right_near_serve_line
    # g_left_far_serve_line
    # h_right_far_serve_line
    # i_center_line_far
    # j_net_line_left
    # k_center_line_right
    # l_net_line_right

    # m_top_front_left
    # n_top_front_right
    # o_top_back_left
    # p_top_back_right
    # q_top_net_line_left
    # r_top_net_line_right


corners_world = {
    "a_front_left": PadelCourt.front_left,
    "b_front_right": PadelCourt.front_right,
    "c_back_left": PadelCourt.back_left,
    "d_back_right": PadelCourt.back_right,
    "e_left_near_serve_line": PadelCourt.left_near_serve_line,
    "f_right_near_serve_line": PadelCourt.right_near_serve_line,
    "g_left_far_serve_line": PadelCourt.left_far_serve_line,
    "h_right_far_serve_line": PadelCourt.right_far_serve_line,
    "i_center_line_far": PadelCourt.center_line[0].flatten().tolist(),
    "j_net_line_left": PadelCourt.net_line[0].flatten().tolist(),
    "k_center_line_right": PadelCourt.center_line[1].flatten().tolist(),
    "l_net_line_right": PadelCourt.net_line[1].flatten().tolist(),
}
corners_world_3d = {
    "a_front_left": (*PadelCourt.front_left, 0.0),
    "b_front_right": (*PadelCourt.front_right, 0.0),
    "c_back_left": (*PadelCourt.back_left, 0.0),
    "d_back_right": (*PadelCourt.back_right, 0.0),
    "e_left_near_serve_line": (*PadelCourt.left_near_serve_line, 0.0),
    "f_right_near_serve_line": (*PadelCourt.right_near_serve_line, 0.0),
    "g_left_far_serve_line": (*PadelCourt.left_far_serve_line, 0.0),
    "h_right_far_serve_line": (*PadelCourt.right_far_serve_line, 0.0),
    "i_center_line_far": (*PadelCourt.center_line[0].flatten().tolist(), 0.0),
    "j_net_line_left": (*PadelCourt.net_line[0].flatten().tolist(), 0.0),
    "k_center_line_near": (*PadelCourt.center_line[1].flatten().tolist(), 0.0),
    "l_net_line_right": (*PadelCourt.net_line[1].flatten().tolist(), 0.0),
    "m_top_front_left": (*PadelCourt.front_left, PadelCourt.backwall_height),
    "n_top_front_right": (*PadelCourt.front_right, PadelCourt.backwall_height),
}
corners_frontwall_world_n = {
    "a_front_left": PadelCourt.front_left_vertical_plane_n,
    "b_front_right": PadelCourt.front_right_vertical_plane_n,
    "m_top_front_left": PadelCourt.top_front_left_vertical_plane_n,
    "n_top_front_right": PadelCourt.top_front_right_vertical_plane_n,
}

corners_world_n = {
    "a_front_left": PadelCourt.front_left_n,
    "b_front_right": PadelCourt.front_right_n,
    "c_back_left": PadelCourt.back_left_n,
    "d_back_right": PadelCourt.back_right_n,
    "e_left_near_serve_line": PadelCourt.left_near_serve_line_n,
    "f_right_near_serve_line": PadelCourt.right_near_serve_line_n,
    "g_left_far_serve_line": PadelCourt.left_far_serve_line_n,
    "h_right_far_serve_line": PadelCourt.right_far_serve_line_n,
    "i_center_line_far": PadelCourt.center_line_n[0].flatten().tolist(),
    "j_net_line_left": PadelCourt.net_line_n[0].flatten().tolist(),
    "k_center_line_near": PadelCourt.center_line_n[1].flatten().tolist(),
    "l_net_line_right": (*PadelCourt.net_line_n[1].flatten().tolist(),),
}

corners_world_3d_n = {
    "a_front_left": (*PadelCourt.front_left_n, 0.0),
    "b_front_right": (*PadelCourt.front_right_n, 0.0),
    "c_back_left": (*PadelCourt.back_left_n, 0.0),
    "d_back_right": (*PadelCourt.back_right_n, 0.0),
    "e_left_near_serve_line": (*PadelCourt.left_near_serve_line_n, 0.0),
    "f_right_near_serve_line": (*PadelCourt.right_near_serve_line_n, 0.0),
    "g_left_far_serve_line": (*PadelCourt.left_far_serve_line_n, 0.0),
    "h_right_far_serve_line": (*PadelCourt.right_far_serve_line_n, 0.0),
    "i_center_line_far": (*PadelCourt.center_line_n[0].flatten().tolist(), 0.0),
    "j_net_line_left": (*PadelCourt.net_line_n[0].flatten().tolist(), 0.0),
    "k_center_line_near": (*PadelCourt.center_line_n[1].flatten().tolist(), 0.0),
    "l_net_line_right": (*PadelCourt.net_line_n[1].flatten().tolist(), 0.0),
}


def get_coords_world_3d():
    return np.array(
        [
            corners_world_3d["a_front_left"],
            corners_world_3d["b_front_right"],
            corners_world_3d["c_back_left"],
            corners_world_3d["d_back_right"],
            corners_world_3d["e_left_near_serve_line"],
            corners_world_3d["f_right_near_serve_line"],
            corners_world_3d["g_left_far_serve_line"],
            corners_world_3d["h_right_far_serve_line"],
        ],
        dtype=np.float32,
    )


def get_coords_world_3d_n():
    return np.array(
        [
            corners_world_3d_n["a_front_left"],
            corners_world_3d_n["b_front_right"],
            corners_world_3d_n["c_back_left"],
            corners_world_3d_n["d_back_right"],
            corners_world_3d_n["e_left_near_serve_line"],
            corners_world_3d_n["f_right_near_serve_line"],
            corners_world_3d_n["g_left_far_serve_line"],
            corners_world_3d_n["h_right_far_serve_line"],
        ],
        dtype=np.float32,
    )


from collections import defaultdict
from functools import partial

import numpy as np
import torch
from kornia.geometry.homography import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
)


def project_points_to_base_plane(points: torch.tensor, H: torch.tensor) -> torch.tensor:
    if len(points.shape) == 2:
        return convert_points_from_homogeneous(
            convert_points_to_homogeneous(points) @ H.T
        )
    elif len(points.shape) == 3:
        return points @ H.T
    else:
        raise RuntimeError(f"{points.shape=} must be of length 2 or 3.")


def convert_corners_to_vec(corners: dict) -> dict:
    """Convert `corners_world_xx_n` to a dict of vectors

    Args:
        corners (dict):

    Returns:
        dict: dict with keys x, y, z and numpy array of each
    """
    vec_of_positions = defaultdict(partial(np.ndarray, 0))
    for _, x_y_z in corners.items():
        for axis, value in zip(["x", "y", "z"], x_y_z, strict=False):
            vec_of_positions[axis] = np.append(vec_of_positions[axis], value)
    return vec_of_positions


def convert_corners_to_coords(corners: dict) -> np.ndarray:
    """Convert `corners_world_xx_n` to a numpy array of shape (12,)

    Args:
        corners (dict):

    Returns:
        np.ndarray: numpy array of shape (12,)
    """
    vec_of_positions = convert_corners_to_vec(corners=corners)
    if "z" in vec_of_positions:
        return np.array(
            [
                (x, y, z)
                for x, y, z in zip(
                    vec_of_positions["x"], vec_of_positions["y"], vec_of_positions["z"]
                )
            ],
            dtype=np.float32,
        )
    return np.array(
        [(x, y) for x, y in zip(vec_of_positions["x"], vec_of_positions["y"])],
        dtype=np.float32,
    )
    # return np.array([vec_of_positions["x"], vec_of_positions["y"]], dtype=np.float32).reshape(-1, 2)


def convert_corners_to_lines(corners: dict):
    sorted_corners = dict(sorted(corners.items()))
    vec = convert_corners_to_vec(corners=sorted_corners)

    xs = np.array(
        [
            (vec["x"][0], vec["x"][1]),  # Back line
            (vec["x"][2], vec["x"][3]),  # Front line
            (vec["x"][4], vec["x"][5]),  # Back serve line
            (vec["x"][6], vec["x"][7]),  # Front serve line
            (vec["x"][0], vec["x"][2]),  # Left side line
            (vec["x"][1], vec["x"][3]),  # Right side line
            (vec["x"][9], vec["x"][8]),  # Center side line
            (vec["x"][10], vec["x"][11]),  # Center side line
        ]
    )
    ys = np.array(
        [
            (vec["y"][0], vec["y"][1]),
            (vec["y"][2], vec["y"][3]),
            (vec["y"][4], vec["y"][5]),
            (vec["y"][6], vec["y"][7]),
            (vec["y"][0], vec["y"][2]),
            (vec["y"][1], vec["y"][3]),
            (vec["y"][9], vec["y"][8]),
            (vec["y"][10], vec["y"][11]),
        ]
    )
    if "z" in vec:
        zs = np.array(
            [
                (vec["z"][0], vec["z"][1]),
                (vec["z"][2], vec["z"][3]),
                (vec["z"][4], vec["z"][5]),
                (vec["z"][6], vec["z"][7]),
                (vec["z"][0], vec["z"][2]),
                (vec["z"][1], vec["z"][3]),
                (vec["z"][9], vec["z"][8]),
                (vec["z"][10], vec["z"][11]),
            ]
        )

    return {"xs": xs, "ys": ys, "zs": zs}


from pathlib import Path


def get_corners_frontwall_image(file_name: str) -> dict:
    file_path = Path(file_name)
    frame_name = "/".join([file_path.parent.name, file_path.stem])
    annotated_images = {
        "curated_001/curated_001_frame_0001": {
            "a_front_left": ((11.11111111111111 / 100.0), (87.90123456790124 / 100.0)),
            "b_front_right": ((89.44444444444444 / 100.0), (88.64197530864197 / 100.0)),
            "m_top_front_left": (
                (8.781544542793732) / 100.0,
                (48.94205883861235) / 100.0,
            ),
            "n_top_front_right": (
                (91.41485490658893) / 100.0,
                (48.850634147450556) / 100.0,
            ),
        }
    }
    return annotated_images[frame_name]


def get_corners_image(file_name: str) -> dict:
    file_path = Path(file_name)
    frame_name = "/".join([file_path.parent.name, file_path.stem])
    annotated_images = {
        "curated_001/curated_001_frame_0001": {
            "a_front_left": ((11.11111111111111 / 100.0), (87.90123456790124 / 100.0)),
            "b_front_right": ((89.44444444444444 / 100.0), (88.64197530864197 / 100.0)),
            "c_back_left": ((31.38888888888889 / 100.0), (29.135802469135804 / 100.0)),
            "d_back_right": ((69.16666666666667 / 100.0), (29.382716049382715 / 100.0)),
            "e_left_near_serve_line": (
                (16.61237785016286 / 100.0),
                (72.58687258687259 / 100.0),
            ),
            "f_right_near_serve_line": (
                (84.14766558089035 / 100.0),
                (72.77992277992279 / 100.0),
            ),
            "g_left_far_serve_line": (
                (29.641693811074916 / 100.0),
                (34.36293436293436 / 100.0),
            ),
            "h_right_far_serve_line": (
                (70.96774193548387 / 100.0),
                (34.387351778656125 / 100.0),
            ),
        }
    }
    return annotated_images[frame_name]


import cv2

from courtvision.vis import load_timg


def compute_homography(annotated_frame, src_corners_n, dst_corners_n):
    src_img_t = load_timg(annotated_frame)
    src_img = src_img_t.squeeze(0).numpy().transpose(1, 2, 0)
    src_img_height, src_img_width, _ = src_img.shape
    dst_points = torch.tensor(
        [
            (x, PadelCourt.length - y)
            for x, y in zip(
                convert_corners_to_vec(dst_corners_n)["x"] * PadelCourt.width,
                convert_corners_to_vec(dst_corners_n)["y"] * PadelCourt.length,
            )
        ]
    )
    src_points = torch.tensor(
        [
            (x, y)
            for x, y in zip(
                convert_corners_to_vec(src_corners_n)["x"] * src_img_width,
                convert_corners_to_vec(src_corners_n)["y"] * src_img_height,
            )
        ]
    )
    if dst_points.shape != src_points.shape:
        raise AssertionError(f"{dst_points.shape=} msut equal {src_points.shape=}")
    homography, _ = cv2.findHomography(src_points.numpy(), dst_points.numpy())
    # TODO: compute distortion and intrnics
    return homography, None, None


def compute_homography_to_vertical_plane(annotated_frame, src_corners_n, dst_corners_n):
    src_img_t = load_timg(annotated_frame)
    src_img = src_img_t.squeeze(0).numpy().transpose(1, 2, 0)
    src_img_height, src_img_width, _ = src_img.shape
    dst_points = torch.tensor(
        [
            (x, y)
            for x, y in zip(
                convert_corners_to_vec(dst_corners_n)["x"] * PadelCourt.width,
                convert_corners_to_vec(dst_corners_n)["y"] * PadelCourt.backwall_height,
            )
        ]
    )
    src_points = torch.tensor(
        [
            (x, y)
            for x, y in zip(
                convert_corners_to_vec(src_corners_n)["x"] * src_img_width,
                convert_corners_to_vec(src_corners_n)["y"] * src_img_height,
            )
        ]
    )
    if dst_points.shape != src_points.shape:
        raise AssertionError(f"{dst_points.shape=} msut equal {src_points.shape=}")
    homography, _ = cv2.findHomography(src_points.numpy(), dst_points.numpy())
    # TODO: compute distortion and intrnics
    return homography, None, None
