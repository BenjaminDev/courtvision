from dataclasses import dataclass

import numpy as np

court_scale = 1


@dataclass
class PadelCourt:
    # REF: https://www.lta.org.uk/4ad2a4/siteassets/play/padel/file/lta-padel-court-guidance.pdf
    width: float = 10.0 * court_scale
    length: float = 20.0 * court_scale
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
        return (0.0, cls.length)

    @classmethod
    @property
    def front_right(cls):
        return (cls.width, cls.length)

    @classmethod
    @property
    def back_left(cls):
        return (0.0, 0.0)

    @classmethod
    @property
    def back_right(cls):
        return (cls.width, 0.0)

    @classmethod
    @property
    def left_near_serve_line(cls):
        return (0.0, cls.length - cls.serve_line_from_back_line)

    @classmethod
    @property
    def right_near_serve_line(cls):
        return (cls.width, cls.length - cls.serve_line_from_back_line)

    @classmethod
    @property
    def left_far_serve_line(cls):
        return (0.0, cls.serve_line_from_back_line)

    @classmethod
    @property
    def right_far_serve_line(cls):
        return (cls.width, cls.serve_line_from_back_line)

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
        return (0.0, cls.length / cls.length)

    @classmethod
    @property
    def front_right_n(cls):
        return (cls.width / cls.width, cls.length / cls.length)

    @classmethod
    @property
    def back_left_n(cls):
        return (0.0, 0.0)

    @classmethod
    @property
    def back_right_n(cls):
        return (cls.width / cls.width, 0.0)

    @classmethod
    @property
    def left_near_serve_line_n(cls):
        return (0.0, (cls.length - cls.serve_line_from_back_line) / cls.length)

    @classmethod
    @property
    def right_near_serve_line_n(cls):
        return (
            cls.width / cls.width,
            (cls.length - cls.serve_line_from_back_line) / cls.length,
        )

    @classmethod
    @property
    def left_far_serve_line_n(cls):
        return (0.0, (cls.serve_line_from_back_line) / cls.length)

    @classmethod
    @property
    def right_far_serve_line_n(cls):
        return (cls.width / cls.width, cls.serve_line_from_back_line / cls.length)


corners_world = {
    "a_front_left": PadelCourt.front_left,
    "b_front_right": PadelCourt.front_right,
    "c_back_left": PadelCourt.back_left,
    "d_back_right": PadelCourt.back_right,
    "e_left_near_serve_line": PadelCourt.left_near_serve_line,
    "f_right_near_serve_line": PadelCourt.right_near_serve_line,
    "g_left_far_serve_line": PadelCourt.left_far_serve_line,
    "h_right_far_serve_line": PadelCourt.right_far_serve_line,
    "i_center_line_left": PadelCourt.center_line[0].flatten().tolist(),
    "j_net_line_left": PadelCourt.net_line[0].flatten().tolist(),
    "i_center_line_right": PadelCourt.center_line[1].flatten().tolist(),
    "j_net_line_right": PadelCourt.net_line[1].flatten().tolist(),
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
    "i_center_line_left": (*PadelCourt.center_line[0].flatten().tolist(), 0.0),
    "j_net_line_left": (*PadelCourt.net_line[0].flatten().tolist(), 0.0),
    "i_center_line_left": (*PadelCourt.center_line[0].flatten().tolist(), 0.0),
    "j_net_line_left": (*PadelCourt.net_line[0].flatten().tolist(), 0.0),
    "i_center_line_rigth": (*PadelCourt.center_line[1].flatten().tolist(), 0.0),
    "j_net_line_right": (*PadelCourt.net_line[1].flatten().tolist(), 0.0),
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
    "i_center_line_left": PadelCourt.center_line_n[0].flatten().tolist(),
    "j_net_line_left": PadelCourt.net_line_n[0].flatten().tolist(),
    "i_center_line_right": PadelCourt.center_line_n[1].flatten().tolist(),
    "j_net_line_right": (*PadelCourt.net_line_n[1].flatten().tolist(),),
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
    "i_center_line_left": (*PadelCourt.center_line_n[0].flatten().tolist(), 0.0),
    "j_net_line_left": (*PadelCourt.net_line_n[0].flatten().tolist(), 0.0),
    "i_center_line_rigth": (*PadelCourt.center_line_n[1].flatten().tolist(), 0.0),
    "j_net_line_right": (*PadelCourt.net_line_n[1].flatten().tolist(), 0.0),
}


from collections import defaultdict
from functools import partial

import numpy as np


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
