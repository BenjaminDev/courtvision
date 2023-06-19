from dataclasses import dataclass

import numpy as np

court_scale = 10


@dataclass
class PadelCourt:
    # REF: https://www.lta.org.uk/4ad2a4/siteassets/play/padel/file/lta-padel-court-guidance.pdf
    width: float = 10.0 * court_scale
    length: float = 20.0 * court_scale
    backwall_height: float = 3.0 * court_scale
    backall_fence_height: float = 4.0 * court_scale
    serve_line_from_back_line: float = 3.0 * court_scale
    line_width: float = 0.05 * court_scale
    net_height: float = 0.78 * court_scale  # 0.78m

    @classmethod
    @property
    def center_line(cls) -> np.array:
        return np.array(
            [
                (cls.width / 2, cls.length - cls.serve_line_from_back_line),
                (cls.width / 2, cls.serve_line_from_back_line),
            ],
            dtype=np.int32,
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
    def m_top_front_left(cls):
        # TODO: add thes
        raise NotImplementedError()

    @classmethod
    @property
    def n_top_front_right(cls):
        raise NotImplementedError()

    @classmethod
    @property
    def o_top_back_left(cls):
        raise NotImplementedError()

    @classmethod
    @property
    def p_top_back_right(cls):
        raise NotImplementedError()

    @classmethod
    @property
    def q_top_net_line_left(cls):
        raise NotImplementedError()

    @classmethod
    @property
    def r_top_net_line_right(cls):
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
    "o_top_back_left": (*PadelCourt.back_left, PadelCourt.backwall_height),
    "p_top_back_right": (*PadelCourt.back_right, PadelCourt.backwall_height),
    "q_top_net_line_left": (
        *PadelCourt.net_line[0].flatten().tolist(),
        PadelCourt.net_height,
    ),
    "r_top_net_line_right": (
        *PadelCourt.net_line[1].flatten().tolist(),
        PadelCourt.net_height,
    ),
    "s_top_net_center": (
        PadelCourt.width / 2,
        PadelCourt.length / 2,
        PadelCourt.net_height,
    ),
    "t_center_center": (PadelCourt.width / 2, PadelCourt.length / 2, 0),
    "u_topfence_front_left": (0, 0, PadelCourt.backall_fence_height),
    "v_topfence_front_right": (PadelCourt.width, 0, PadelCourt.backall_fence_height),
    "w_topfence_back_left": (0, PadelCourt.length, PadelCourt.backall_fence_height),
    "x_topfence_back_right": (
        PadelCourt.width,
        PadelCourt.length,
        PadelCourt.backall_fence_height,
    ),
    "y_top_center_left": (0, PadelCourt.length / 2, PadelCourt.backwall_height),
    "z_top_center_right": (
        PadelCourt.width,
        PadelCourt.length / 2,
        PadelCourt.backwall_height,
    ),
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
from typing import Literal


def get_corners_verital_plane_on_image(
    file_name: str, plane: Literal["front", "back"]
) -> dict:
    file_path = Path(file_name)
    frame_name = "/".join([file_path.parent.name, file_path.stem])
    annotated_images = {
        "curated_001/curated_001_frame_0001": {
            "a_front_left": ((11.11111111111111 / 100.0), (87.90123456790124 / 100.0)),
            "b_front_right": ((89.44444444444444 / 100.0), (88.64197530864197 / 100.0)),
            "c_back_left": ((31.18673582738404) / 100.0, (29.00062932223646) / 100.0),
            "d_back_right": ((68.65158264621417) / 100.0, (28.943474276940947) / 100.0),
            "m_top_front_left": (
                (8.781544542793732) / 100.0,
                (48.94205883861235) / 100.0,
            ),
            "n_top_front_right": (
                (91.41485490658893) / 100.0,
                (48.850634147450556) / 100.0,
            ),
            "o_top_back_left": (
                (30.672799144493634) / 100.0,
                (8.698873476649666) / 100.0,
            ),
            "p_top_back_right": (
                (69.66060228055649) / 100.0,
                (8.748200961152353) / 100.0,
            ),
        }
    }
    points = annotated_images[frame_name]
    if plane == "front":
        for k in ["c_back_left", "d_back_right", "o_top_back_left", "p_top_back_right"]:
            points.pop(k)
    elif plane == "back":
        for k in [
            "a_front_left",
            "b_front_right",
            "m_top_front_left",
            "n_top_front_right",
        ]:
            points.pop(k)
    elif plane == "both":
        pass
    else:
        raise ValueError(f"{plane=} must be one of 'front', 'back' or 'both'")
    return points


def get_planar_points_padel_court(
    available_labels: set[str], minimal_set_count: int
) -> list[set]:
    # floor plane points
    available_planes_for_calibration = []
    floor_plane_points = {
        "a_front_left",
        "b_front_right",
        "c_back_left",
        "d_back_right",
        "e_left_near_serve_line",
        "f_right_near_serve_line",
        "g_left_far_serve_line",
        "h_right_far_serve_line",
        "i_center_line_far",
        "j_net_line_left",
        "k_center_line_near",
        "l_net_line_right",
        "t_center_center",
    }
    left_vertical_plane_points = {
        "a_front_left",
        "c_back_left",
        "e_left_near_serve_line",
        "g_left_far_serve_line",
        "j_net_line_left",
        "m_top_front_left",
        "o_top_back_left",
        "q_top_net_line_left",
        "u_topfence_front_left",
        "w_topfence_back_left",
    }

    right_vertical_plane_points = {
        "b_front_right",
        "d_back_right",
        "f_right_near_serve_line",
        "h_right_far_serve_line",
        "n_top_front_right",
        "p_top_back_right",
        "r_top_net_line_right",
        "v_topfence_front_right",
        "x_topfence_back_right",
        "z_top_center_right",
        "l_net_line_right",
    }
    front_vertical_plane_points = {
        "a_front_left",
        "b_front_right",
        "m_top_front_left",
        "n_top_front_right",
        "u_topfence_front_left",
        "v_topfence_front_right",
    }
    back_vertical_plane_points = {
        "c_back_left",
        "d_back_right",
        "o_top_back_left",
        "p_top_back_right",
        "w_topfence_back_left",
        "x_topfence_back_right",
    }

    center_vertical_plane_points = {
        "j_net_line_left",
        "q_top_net_line_left",
        "r_top_net_line_right",
        "l_net_line_right",
        # "t_center_center",
    }
    top_horizontal_plane_points = {
        "m_top_front_left",
        "n_top_front_right",
        "o_top_back_left",
        "p_top_back_right",
        "y_top_center_left",
        "z_top_center_right",
    }
    topfence_horizontal_plane_points = {
        "u_topfence_front_left",
        "v_topfence_front_right",
        "w_topfence_back_left",
        "x_topfence_back_right",
    }
    all_planar_sets = [
        floor_plane_points,
        left_vertical_plane_points,
        right_vertical_plane_points,
        front_vertical_plane_points,
        back_vertical_plane_points,
        center_vertical_plane_points,
        top_horizontal_plane_points,
        topfence_horizontal_plane_points,
    ]
    # from itertools import combinations
    # from itertools import chain
    # selected_available_labels = [set(o) for  o in chain.from_iterable(
    #     set(combinations(available_labels, o))
    #     for o in range(
    #             minimal_set_count,
    #             minimal_set_count +1,
    #             # len(available_labels) + 1,
    #     )
    # )]

    # for selected_labels in selected_available_labels:
    for planar_set in all_planar_sets:
        if len(available_labels.intersection(planar_set)) >= minimal_set_count:
            available_planes_for_calibration.append(
                available_labels.intersection(planar_set)
            )

    return available_planes_for_calibration


from typing import Optional


def get_planar_point_correspondences(
    world_points: dict[str, tuple[float, float]],
    image_points: dict[str, tuple[float, float]],
    available_labels: Optional[set[str]] = None,
    minimal_set_count: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Given a set of named points in the world and image, return a list of point correspondences
    where all points are coplanar.
    If a specified set `available_labels` is given, only return point correspondences where all
    points are in that set.
    Args:
        world_points (dict[str, tuple[float, float]]): Dict of named points in the world coordinate frame.
        image_points (dict[str, tuple[float, float]]): Dict of named points in the image coordinate frame.
        available_labels (Optional[set[str]], optional): Set of labels to use if None all labels are used. Defaults to None.
        minimal_set_count (int, optional): Sets of corresponding points . Defaults to 4.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Returns a list of point correspondences where all points are coplanar.
        list[tuple[Nx3, Nx2]]
    """
    available_labels = available_labels or set(image_points.keys())
    available_planes_for_calibration = get_planar_points_padel_court(
        available_labels=available_labels,
        minimal_set_count=minimal_set_count,
    )
    from courtvision.data import dict_to_points

    planar_point_correspondences = []
    for plane in available_planes_for_calibration:
        world_points_on_plane_dict = {k: world_points[k] for k in plane}
        image_points_on_plane_dict = {k: image_points[k] for k in plane}
        world_points_on_plane, _ = dict_to_points(world_points_on_plane_dict)
        image_points_on_plane, _ = dict_to_points(image_points_on_plane_dict)
        planar_point_correspondences.append(
            (world_points_on_plane, image_points_on_plane)
        )
    return planar_point_correspondences


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
            "i_center_line_far": (
                (49.974415820476416 / 100.0),
                (33.73523908457271 / 100.0),
            ),
            "j_net_line_left": (
                (24.39366213014177 / 100.0),
                (49.3124355099835 / 100.0),
            ),
            "k_center_line_near": (
                (50.16556291390729 / 100.0),
                (72.05882352941177 / 100.0),
            ),
            "l_net_line_right": (
                (76.01445288204798 / 100.0),
                (47.96259686296029 / 100.0),
            ),
            "m_top_front_left": (
                (8.781544542793732 / 100.0),
                (48.942058838612354 / 100.0),
            ),
            "n_top_front_right": (
                (91.41485490658893 / 100.0),
                (48.850634147450556 / 100.0),
            ),
            "o_top_back_left": (
                (30.672799144493634 / 100.0),
                (8.698873476649666 / 100.0),
            ),
            "p_top_back_right": (
                (69.48541987969993 / 100.0),
                (8.732922374525746 / 100.0),
            ),
            "q_top_net_line_left": (
                (24.142083082480436 / 100.0),
                (40.64171122994652 / 100.0),
            ),
            "r_top_net_line_right": (
                (75.91812161348585 / 100.0),
                (40.106951871657756 / 100.0),
            ),
            "s_top_net_center": (
                (50.175438596491226 / 100.0),
                (41.12149532710281 / 100.0),
            ),
            "t_center_center": (
                (50.175438596491226 / 100.0),
                (47.97507788161994 / 100.0),
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
        raise AssertionError(f"{dst_points.shape=} must equal {src_points.shape=}")
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
    cv2.findHomography
    # TODO: compute distortion and intrnics
    return homography, None, None


def solve_for_camera_matrix(
    world_points: torch.Tensor,
    image_points: torch.Tensor,
    image_size: tuple[int, int],
    repo_erro_threshold: float = 1e-1,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """From a set of world points and image points, solve for the camera matrix and distortion coefficients.
    Note: All world points must have the same z value. i.e lie on the same plane.

    Args:
        world_points (torch.Tensor): Tensor of world points.
        image_points (torch.Tensor): Tensor of image points.
        image_size (tuple[int, int]): Image dimensions as (Width, Height).
        repo_error (float, optional): Reprojection error measured in pixels. Defaults to 1e-1.

    Returns (Tuple[torch.Tensor, torch.Tensor, float]): camera_matrix (3x3), dist_coeffs (1x5), repo_erro

    """
    if len(world_points.shape) == 3:
        _world_points = [world_points.squeeze(0).numpy().astype(np.float32)]
    elif len(world_points.shape) == 2:
        _world_points = [world_points.numpy().astype(np.float32)]
    else:
        raise RuntimeError(f"{world_points.shape=} must be of length 2 or 3.")
    if len(image_points.shape) == 3:
        _image_points = [image_points.squeeze(0).numpy().astype(np.float32)]
    elif len(image_points.shape) == 2:
        _image_points = [image_points.numpy().astype(np.float32)]
    else:
        raise RuntimeError(f"{image_points.shape=} must be of length 2 or 3.")

    # if not all(o[-1] == _world_points[0][0][-1] for o in _world_points[0]):
    # raise RuntimeError(f"{_world_points=} must have same z value")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)

    repo_erro, camera_matrix, dist_coeffs, *_ = cv2.calibrateCamera(
        objectPoints=_world_points,
        imagePoints=_image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        criteria=criteria,
    )
    if repo_erro > repo_erro_threshold:
        raise RuntimeError(f"{repo_erro=} must be less than 1e-6")
    print(f"{repo_erro=}")
    return camera_matrix, dist_coeffs, repo_erro


# dist_coeffs
def solve_for_projection_matrix_v2(
    world_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    in_object_coordinate_frame: bool = True,
):
    import kornia

    extrinsics = (
        kornia.geometry.calibration.solve_pnp_dlt(
            world_points=torch.tensor(world_points).unsqueeze(0),
            img_points=torch.tensor(image_points).unsqueeze(0),
            intrinsics=torch.tensor(camera_matrix).unsqueeze(0),
        )
        .squeeze(0)
        .numpy()
    )
    print(f"{extrinsics[:3, :3]=}")
    print(f"{extrinsics[:3, 3]=}")
    rvec, _ = cv2.Rodrigues(extrinsics[:3, :3])
    tvec = extrinsics[:3, 3]
    reprojected_image_points, _ = cv2.projectPoints(
        world_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    reprojected_image_points = reprojected_image_points.reshape(-1, 2)
    reprojection_error = np.linalg.norm(
        reprojected_image_points - image_points, axis=1
    ).mean()
    print(f"{reprojection_error=}")

    return rvec, tvec, reprojected_image_points


def solve_for_projection_matrix(
    world_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    in_object_coordinate_frame: bool = True,
):
    success, rvec, tvec = cv2.solvePnP(
        world_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        useExtrinsicGuess=False,
    )
    inliners = None
    if not success:
        raise RuntimeError(f"{success=} Failed to compute projection matrix")
    # success, rvec, tvec, inliners = cv2.solvePnPRansac(
    #     world_points,
    #     image_points,
    #     camera_matrix,
    #     dist_coeffs,
    #     rvec=rvec,
    #     tvec=tvec,
    #     flags=cv2.SOLVEPNP_ITERATIVE,
    #     useExtrinsicGuess=True,
    #     # confidence=0.01,
    #     # reprojectionError=8,
    # )
    print(f"{inliners=}")
    reprojected_image_points, _ = cv2.projectPoints(
        world_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    reprojected_image_points = reprojected_image_points.reshape(-1, 2)
    reprojection_error = np.linalg.norm(
        reprojected_image_points - image_points, axis=1
    ).mean()
    print(f"{reprojection_error=}")
    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    # rmat[0, :] *= -1
    return rmat, rvec, tvec, reprojected_image_points
    # R, t = cv2.Rodrigues(rvec)
    # T = np.append(t, [1], axis=0)
    # transformation_matrix = np.dot(R, T)
    # return transformation_matrix
    # Transform the object points from the object coordinate frame to the camera coordinate frame.
    # camera_points = np.dot(transformation_matrix, object_points.T).T
    if in_object_coordinate_frame:
        # Change rotation matrix to be in the object coordinate frame
        rmat_inv = np.linalg.inv(rmat)
        # rmat_inv = rmat.T
        # tvec_inv = -1*tvec
        # return rmat_inv
        # Compute inverse translation
        t_inv = -np.dot(rmat_inv, tvec)
        projection_matrix = np.hstack((rmat_inv, t_inv))
    else:
        # raise NotImplementedError()
        # return rmat
        # Concatenate rotation matrix and translation vector to create a 3x4 transformation matrix
        projection_matrix = np.hstack((rmat, tvec))
    return projection_matrix


def transfrom_points(
    points: np.ndarray, transformation_matrix: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    return np.dot(transformation_matrix.T, points.T + tvec).T
    # return (transformation_matrix @ points.T).T - tvec.T
    # return (transformation_matrix @ (points.T-tvec)).T


def transform_points_inverse(
    points: np.ndarray, transformation_matrix: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    return (transformation_matrix.T @ (points + tvec).T).T

    # return np.dot(transformation_matrix, points.T).T


def denormalize_points(named_points, width, height):
    points = np.array([(v[0] * width, v[1] * height) for _, v in named_points.items()])
    return points


from typing import TypeVar

Point2D = TypeVar("Point2D", np.array, tuple, list)


def denormalize_as_named_points(
    normalised_named_points: dict[str, Point2D], image_width: int, image_height: int
) -> dict[str, Point2D]:
    """Transforms a dict of normalized points `0 to 1` to image points using the
    supplied image dimension.

    Args:
        normalised_named_points (dict[str, Point2D]): Dict of points normalised from `0.0` to `1.0`
        image_width (int): Image width to expand to.
        image_height (int): Image height to expand to.

    Returns:
        dict[str, Point2D]: Retruns a dict of similar struture but with image points.
    """
    return {
        k: (v[0] * image_width, v[1] * image_height)
        for k, v in normalised_named_points.items()
    }


def convert_obj_points_to_planar(object_points: np.array) -> np.array:
    """Converts object points to planar points by finding the common axis and permuting the points so that the common axis is the last axis.
    Assumes that the object points are planar.

    Args:
        object_points (np.array): _description_

    Raises:
        ValueError: When points are not planar

    Returns:
        np.array: _description_
    """
    common_axis = None
    for axis in [0, 1, 2]:
        if all(object_points[0, axis] == o for o in object_points[:, axis]):
            common_axis = axis
            break
    if common_axis is None:
        raise ValueError("Could not find common axis")
    # permute the object points so that the common axis is the last axis
    return np.concatenate(
        [
            object_points[:, [i for i in range(3) if i != common_axis]],
            np.zeros((object_points.shape[0], 1)),
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class CameraInfo:
    valid_for_clip_ids: set[str]
    camera_matrix: np.array
    distortion_coefficients: np.array
    rotation_vector: np.array
    translation_vector: np.array
    image_width: int
    image_height: int
    error_in_reprojecred_planar_points: float
    error_in_reprojecred_points: float

    def save(self, file_name: str):
        import numpy as np

        np.savez(
            file_name,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.distortion_coefficients,
            rotation_vector=self.rotation_vector,
            translation_vector=self.translation_vector,
            image_width=self.image_width,
            image_height=self.image_height,
            error_in_reprojecred_planar_points=self.error_in_reprojecred_planar_points,
            error_in_reprojecred_points=self.error_in_reprojecred_points,
            valid_for_clip_ids=self.valid_for_clip_ids,
        )

    @staticmethod
    def load(file_name: str):
        import numpy as np

        data = np.load(file_name, allow_pickle=True)
        return CameraInfo(
            camera_matrix=data["camera_matrix"],
            distortion_coefficients=data["distortion_coefficients"],
            rotation_vector=data["rotation_vector"],
            translation_vector=data["translation_vector"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            error_in_reprojecred_planar_points=data[
                "error_in_reprojecred_planar_points"
            ],
            error_in_reprojecred_points=data["error_in_reprojecred_points"],
            valid_for_clip_ids=data["valid_for_clip_ids"].tolist(),
        )


def calibrate_and_evaluate(
    valid_clip_ids: set[str],
    *,
    calibration_correspondences_selected,
    pose_correspondences_selected,
    image_width,
    image_height,
    all_world_points,
    all_image_points,
) -> CameraInfo:
    repo_erro, camera_matrix, dist_coeffs, *_ = cv2.calibrateCamera(
        objectPoints=[
            convert_obj_points_to_planar(obj)
            for obj, _ in calibration_correspondences_selected
        ],
        imagePoints=[img for _, img in calibration_correspondences_selected],
        imageSize=(image_width, image_height),
        cameraMatrix=None,
        distCoeffs=None,
    )
    # print(repo_erro)

    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (image_width, image_height),
        1,
        (image_width, image_height),
        False,
    )

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=np.concatenate([obj for obj, _ in pose_correspondences_selected]),
        imagePoints=np.concatenate([img for _, img in pose_correspondences_selected]),
        cameraMatrix=optimal_camera_matrix,
        distCoeffs=dist_coeffs,
        # None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        useExtrinsicGuess=False,
    )

    # print(f"{success=}")
    reprojected_image_points, _ = cv2.projectPoints(
        all_world_points,
        rvec,
        tvec,
        optimal_camera_matrix,
        dist_coeffs,
    )
    # print(f"{optimal_camera_matrix=}")
    reprojected_image_points = reprojected_image_points.reshape(-1, 2)
    reprojection_error = np.linalg.norm(
        reprojected_image_points - all_image_points, axis=1
    ).mean()
    # print(f"{reprojection_error=}")
    return CameraInfo(
        camera_matrix=optimal_camera_matrix,
        distortion_coefficients=dist_coeffs,
        rotation_vector=rvec,
        translation_vector=tvec,
        image_width=image_width,
        image_height=image_height,
        error_in_reprojecred_planar_points=repo_erro,
        error_in_reprojecred_points=reprojection_error,
        valid_for_clip_ids=valid_clip_ids,
    )


from dataclasses import dataclass
from itertools import chain, combinations, product


def find_optimal_calibration_and_pose(
    valid_clip_ids: set[str],
    calibration_correspondences: list[tuple[np.array, np.array]],
    pose_correspondences: list[tuple[np.array, np.array]],
    image_width: int,
    image_height: int,
    all_image_points: np.array,
    all_world_points: np.array,
) -> CameraInfo:
    """
    Givern a set of calibration and pose correspondences, find the optimal calibration and pose.
    This is done by building up combinations of these sets and evaluating the reprojection error.
    The reprojection error is the mean of the euclidean distance between the reprojected points and the actual points.
    The evvaluation is on all `all_image_points` and `all_world_points`.

    Args:
        valid_clip_ids (set[str]): _description_
        calibration_correspondences (list[tuple[np.array, np.array]]): _description_
        pose_correspondences (list[tuple[np.array, np.array]]): _description_
        image_width (int): Image width
        image_height (int): Image height
        all_image_points (np.array): 3D points that we want to reproject.
        all_world_points (np.array): 2D points that are where we expect the 3D points to be reprojected to.

    Raises:
        RuntimeError: _description_

    Returns:
        CameraInfo: _description_
    """
    CALIBRATION_MIN_PAIRS = 4
    CALIBRATION_MAX_PAIRS = min(8, len(calibration_correspondences))

    POSE_MIN_PAIRS = 4
    POSE_MAX_PAIRS = min(8, len(pose_correspondences))

    calibration_indexes = [o for o in range(len(calibration_correspondences))]
    calibration_selected_pairs: list[tuple[int, ...]] = list(
        chain.from_iterable(
            (combinations(calibration_indexes, num_pairs_to_use))
            for num_pairs_to_use in range(CALIBRATION_MIN_PAIRS, CALIBRATION_MAX_PAIRS)
        )
    )

    pose_indexes = [o for o in range(len(pose_correspondences))]
    pose_selected_pairs: list[tuple[int, ...]] = list(
        chain.from_iterable(
            (combinations(pose_indexes, num_pairs_to_use))
            for num_pairs_to_use in range(POSE_MIN_PAIRS, POSE_MAX_PAIRS)
        )
    )

    best_error_in_reprojecred_points = 10000.0
    best_camera_info = None

    for calibration_pair, pose_pair in product(
        calibration_selected_pairs, pose_selected_pairs
    ):
        calibration_correspondences_selection = [
            calibration_correspondences[o] for o in calibration_pair
        ]
        pose_correspondences_selection = [pose_correspondences[o] for o in pose_pair]

        camera_info = calibrate_and_evaluate(
            valid_clip_ids=valid_clip_ids,
            calibration_correspondences_selected=calibration_correspondences_selection,
            pose_correspondences_selected=pose_correspondences_selection,
            image_width=image_width,
            image_height=image_height,
            all_image_points=all_image_points,
            all_world_points=all_world_points,
        )
        if camera_info.error_in_reprojecred_points < best_error_in_reprojecred_points:
            best_camera_info = camera_info
    if best_camera_info is None:
        raise RuntimeError("Failed to find optimal calibration and pose")
    return best_camera_info


if __name__ == "__main__":
    import rerun as rr

    rr.init("geometry", spawn=True)
