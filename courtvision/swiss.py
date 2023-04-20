from pathlib import Path

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
