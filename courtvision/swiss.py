import datetime
import os
import warnings
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy as np


def save_camera_params(
    *, file_name, homography, intrinsic_matrix=None, distortion_coeffs=None
):
    """Save camera parameters to a file."""
    file_path = Path(file_name)
    if intrinsic_matrix is not None:
        np.save(file_path.parent / "intrinsic_matrix", intrinsic_matrix)
    if distortion_coeffs is not None:
        np.save(file_path.parent / "distortion_coeffs", distortion_coeffs)
    np.save(file_path.parent / "homography", homography)


def get_latest_file(dir: Path, file_suffix: str = ".pt") -> Path:
    """Fetch the model_path of the latest model in `model_dir`.

    Args:
        model_dir (Path): path to directory of models.
        model_suffix (str, optional): extention of model format. Defaults to ".pt".

    Returns:
        Path: path to most recent model.
    """

    # Get the most recent file based on modification time
    most_recent_file = None
    most_recent_time = datetime.datetime.min
    for file in dir.glob(f"*{file_suffix}"):
        modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(file))
        if modification_time > most_recent_time:
            most_recent_time = modification_time
            most_recent_file = file
    return most_recent_file


def mark_as_deprecated(
    to_be_removed_in_version: tuple[int, int, int],
    details: str,
    moved_to: Optional[str] = None,
):
    """Marks a function as deprecated.
    Args:
        to_be_removed_in_version (tuple[int, int, int]): after which version the function will be removed.
        details (str): Message for the caller.
        moved_to (Optional[str], optional): If a function exists that callers should migrate to. Defaults to None.

    Returns:
        Callable: function wrapper
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in Version {to_be_removed_in_version}. Details {details}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return inner
