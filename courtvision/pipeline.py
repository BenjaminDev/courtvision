import enum
import itertools
import json
from enum import Enum
from hashlib import md5
from logging import basicConfig, getLogger
from pathlib import Path

import boto3
import cv2
import numpy as np
import rerun as rr
import structlog
import torch
import torchvision
from kornia.geometry import unproject_points
from structlog import wrap_logger

from courtvision.data import (
    Annotation,
    ClipSegmentResult,
    CourtAnnotatedSample,
    CourtVisionArtifacts,
    KeypointValue,
    PadelDataset,
    RectValue,
    StreamType,
    VideoRectValue,
    dict_to_points,
    download_data_item,
    frames_from_clip_segments,
    get_normalized_calibration_image_points_and_clip_ids,
)
from courtvision.geometry import (
    CameraInfo,
    PadelCourt,
    calibrate_and_evaluate,
    convert_obj_points_to_planar,
    corners_world_3d,
    court_scale,
    denormalize_as_named_points,
    find_optimal_calibration_and_pose,
    get_planar_point_correspondences,
    get_planar_points_padel_court,
)
from courtvision.models import BallDetector, PlayerDetector
from courtvision.swiss import get_latest_file, mark_as_deprecated
from courtvision.vis import log_court_layout

logger = structlog.get_logger()


class Verbosity(Enum):
    SILENT = 0
    PROGRESS = 1
    DEBUG = 2


def log_ball_detections(
    detections: dict[str, torch.Tensor],
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
            # score=score.item(),
            # label=label.item(),
            timeless=True,
        )


def point_of_intersection(
    translation_vector: np.array,
    unprojected_point_world_coordinate: np.array,
):
    import numpy as np

    if translation_vector.shape[0] == 3:
        translation_vector = translation_vector.T

    # Calculate direction of the ray from camera's position to the 3D point
    ray_direction = unprojected_point_world_coordinate - translation_vector

    # Define vectors in the plane
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])

    # Compute normal to the plane as cross product of v1 and v2
    normal = np.cross(v1, v2)

    # Assuming the plane passes through the origin
    point_on_plane = np.array([0, 0, 0])

    # Equation of the plane: (r - point_on_plane) . normal = 0
    # Substituting r = tvec + t*ray_direction in the plane equation, we get (tvec + t*ray_direction - point_on_plane) . normal = 0

    # Solving for t
    t = np.dot((point_on_plane - translation_vector), normal) / np.dot(
        ray_direction, normal
    )

    # Compute the intersection point
    intersection = translation_vector + t * ray_direction
    return intersection


def image_point_to_world_court_point(
    image_point: torch.Tensor,
    camera_matrix: torch.Tensor,
    translation_vector: torch.Tensor,
    rotation_vector: torch.Tensor,
):
    import cv2
    import numpy as np
    from torch.nn.functional import pad

    # Ensure the correct shape of input tensors
    # image_point = image_point.view(-1, 1)
    # translation_vector = translation_vector.view(3, 1)
    # rotation_vector = rotation_vector.view(3, 1)
    # # Convert tensors to numpy arrays for processing
    # image_point = image_point.numpy()
    # camera_matrix = camera_matrix.numpy()
    # translation_vector = translation_vector.numpy()
    # rotation_vector = rotation_vector.numpy()
    image_point = image_point.numpy()
    # Convert to homogenous
    point_2D_hom = np.vstack([image_point.T, np.array([[1]])])

    # Apply inverse camera matrix to get ray direction in camera coords.
    ray_direction_cam = np.linalg.inv(camera_matrix) @ point_2D_hom

    # Convert ray direction to world coordinates
    R, _ = cv2.Rodrigues(rotation_vector)
    ray_direction_world = np.dot(R.T, ray_direction_cam)

    # Normalize the ray direction
    ray_direction_world /= np.linalg.norm(ray_direction_world)

    # Define the plane
    # Normal vector for plane (1,0,0) cross (0,1,0) = (0,0,1)
    normal = np.array([[0], [1], [0]])

    # Assuming the plane passes through the origin
    point_on_plane = np.array([[0], [0], [0]])

    # Compute intersection
    numerator = np.dot((point_on_plane - translation_vector).T, normal)
    denominator = np.dot(ray_direction_world.T, normal)
    t = numerator / denominator

    intersection = translation_vector + t * ray_direction_world

    return intersection, ray_direction_world


import ultralytics
from torch.functional import F

from courtvision.geometry import project_points_to_base_plane
from courtvision.vis import colours_per_player_idx

# def image_point_to_world_court_point(
#         image_point: torch.Tensor,
#         camera_matrix: torch.Tensor,
#         translation_vector: torch.Tensor,
#         rotation_vector: torch.Tensor,
# ):
#     import numpy as np
#     import cv2

#     if translation_vector.shape[0] == 3:
#         translation_vector = translation_vector.T
#     # Convert to homogenous
#     point_2D_hom = F.pad(image_point, [0, 1], "constant", 1.0).squeeze(0).numpy()

#     # Apply inverse camera matrix to get ray direction in camera coords.
#     ray_direction_cam = np.linalg.inv(camera_matrix) @ point_2D_hom

#     # Convert ray direction to world coordinates
#     R, _ = cv2.Rodrigues(rotation_vector)
#     ray_direction_world = R.T @ ray_direction_cam

#     # Normalize the ray direction
#     ray_direction_world /= np.linalg.norm(ray_direction_world)

#     # Define the plane
#     # Normal vector for plane (1,0,0) cross (0,1,0) = (0,0,1)
#     normal = np.array([0, 0, 1])

#     # Assuming the plane passes through the origin
#     point_on_plane = np.array([0, 0, 0])

#     # Compute intersection
#     t = np.dot((point_on_plane - translation_vector), normal) / np.dot(ray_direction_world, normal)
#     intersection = translation_vector + t * ray_direction_world
#     return intersection.squeeze(0)


def apply_camera_to_world_transform(
    point_in_camera_space: np.array,
    translation_vector: np.array,
    rotation_vector: np.array,
):
    """transform a point in camera space to world space"""
    import cv2

    if translation_vector.shape[0] == 3:
        translation_vector = translation_vector.T
    if rotation_vector.shape[0] == 3:
        rotation_vector = rotation_vector.T

    R, _ = cv2.Rodrigues(rotation_vector)

    # Convert 3D point to world's coordinate system
    point_3D_world = np.dot(R.T, point_in_camera_space) + translation_vector

    return point_3D_world
    # rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    # T = np.concatenate((rotation_matrix.T, -translation_vector.T), axis=1)

    # point_in_camera_space_homogeneous= F.pad(point_in_camera_space, [0,1], "constant", 1.0)
    # point_in_world_space = np.matmul(T, point_in_camera_space_homogeneous)
    # point_in_world_space = np.matmul(rotation_matrix.T, (point_in_camera_space.T - translation_vector).squeeze(0))
    # point_in_world_space =  point_in_camera_space + translation_vector
    # breakpoint()
    # return point_in_world_space


def camera_space_to_world_space(
    camera_point: np.array,
    translation_vector: np.array,
    rotation_vector: np.array,
):
    if not camera_point.shape[0] == 3:
        raise ValueError("Camera point must be a 3D point with shape (3,)")
    R, _ = cv2.Rodrigues(rotation_vector)
    world_point = (R.T @ (camera_point - translation_vector)).T
    return world_point


def compute_ray_intersecting_plane(
    point_a_on_ray: np.array,
    point_b_on_ray: np.array,
    plane_normal: np.array = np.array([[0, 0, 1]]),
    plane_point: np.array = np.array([0, 0, 0]),
):
    """Given two points on a ray, compute the point of intersection with a plane.
    The plane is defined as a normal vector and a point on the plane.

    Args:
        point_a_on_ray (np.array): 3D point on the ray with shape (3, 1)
        point_b_on_ray (np.array): 3D point on the ray with shape (3, 1)
        plane_normal (np.array, optional): Unit vector pointing out the plane. Defaults to np.array([[0, 0, 1]]). Shape (1, 3)
        plane_point (np.array, optional): Point on the plane. Defaults to np.array([0, 0, 0]). Shape (3,)

    Returns:
        np.array: Returns the 3D point of intersection with shape (3,)
    """
    # Vector along the ray direction A -> B
    if not (point_a_on_ray.shape == point_b_on_ray.shape == (3, 1)):
        raise ValueError(
            f"Point A and B must be 3D points with shape (3, 1). {point_a_on_ray.shape=} {point_b_on_ray.shape=}"
        )
    if not plane_normal.shape == (1, 3) and np.linalg.norm(plane_normal) == 1:
        raise ValueError(
            "Plane normal must be a unit vector with shape (1, 3) and norm 1"
        )
    if not plane_point.shape == (3,):
        raise ValueError("Plane point must be a 3D point with shape (3,)")

    ray_direction = point_b_on_ray - point_a_on_ray
    # Finding parameter t
    t = -(np.dot(plane_normal, point_a_on_ray) + plane_point) / np.dot(
        plane_normal, ray_direction
    )
    # Finding point of intersection
    intersection = (point_a_on_ray.T + t * ray_direction.T).squeeze(0)
    return intersection


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
            f"world/camera/image/Player_{int(idx)}",
            (x1, y1, (x2 - x1), (y2 - y1)),
            color=colours_per_player_idx[int(idx)],
        )
        # Compute the 3D point of the player's feet
        # Use 2 depth values to unproject the point from the image plane to the camera plane
        depths = torch.tensor(
            [[1.0 * court_scale, 20.0 * court_scale]]
        ).T  # Depth values in [mm * court_scale]
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
        # rr.log_points(f"world/camera/Player_{int(idx)}", mid_feet_base_camera_space,  )
        rr.log_point(
            f"world/Player_{int(idx)}",
            intersection,
            radius=player_maker_radius,
            color=colours_per_player_idx[int(idx)],
        )
        # rr.log_arrow(
        #     "world/camera/ray",
        #     origin=translation_vector.squeeze(),
        #     vector=rotation_vector.squeeze(),
        #     # length=100,
        #     width_scale=100.0,
        # )
        # mid_feet_base_3d, ray_direction_world = image_point_to_world_court_point(
        #     image_point=mid_feet,
        #     camera_matrix=camera_matrix,
        #     translation_vector=translation_vector,
        #     rotation_vector=rotation_vector,
        # )
        # rr.log_arrow(
        #     "world/camera/ray_world",
        #     origin=translation_vector.squeeze(),
        #     vector=ray_direction_world.squeeze(),
        #     width_scale=10.0,
        # )
        # from kornia.geometry import unproject_points
        # mid_feet_base_camera_space = unproject_points(point_2d=mid_feet, camera_matrix=camera_matrix,depth=torch.tensor([10.0])).squeeze(0)
        # mid_feet_base = apply_camera_to_world_transform(
        #     mid_feet_base_camera_space,
        #     translation_vector=translation_vector,
        #     rotation_vector=rotation_vector,
        # )
        # mid_feet_base_3d = point_of_intersection(
        #     translation_vector=translation_vector,
        #     unprojected_point_world_coordinate=mid_feet_base.numpy(),
        # ).squeeze(0)

        # mid_feet_base_3d = mid_feet_base
        # (
        # F.pad(mid_feet_base, (0, 1), mode="constant", value=0.0) / 10.0
        # )
        # Switch Y and Z axis
        # x = mid_feet_base_3d[0].item()
        # y = mid_feet_base_3d[1].item()
        # z = mid_feet_base_3d[2].item()
        # mid_feet_base_3d[2] = player_maker_radius
        # mid_feet_base_3d[1] = mid_feet_base_3d[1]
        # mid_feet_base_3d[0] = mid_feet_base_3d[0]
        # # rr.log_point("world/debug",mid_feet_base, radius=10)
        # rr.log_point(
        #     f"world/camera/Player_{int(idx)}",
        #     mid_feet_base_3d,
        #     radius=player_maker_radius,
        #     color=colours_per_player_idx[int(idx)],
        # )


def pipeline(
    artifacts: CourtVisionArtifacts,
):
    # Calibrate camera
    artifacts = calibrate_camera(artifacts)

    # Detect ball and players in the image plane
    # for artifacts.dataset


def calibrate_camera(
    artifacts: CourtVisionArtifacts, verbose: Verbosity = Verbosity.PROGRESS
) -> CourtVisionArtifacts:
    if artifacts.camera_info:
        logger.info(
            "Camera already calibrated - using supplied camera_info",
            camera_info=artifacts.camera_info,
        )
        return artifacts
    if artifacts.camera_info_path.is_file():
        artifacts.camera_info = CameraInfo.load(artifacts.camera_info_path)
        logger.info(
            "Camera already calibrated - using cached version",
            camera_info=artifacts.camera_info,
        )
        return artifacts

    frame, uid = next(
        frames_from_clip_segments(
            artifacts.dataset,
            local_path=artifacts.local_cache_path,
            stream_type=StreamType.VIDEO,
        )
    )

    image_width, image_height = frame["data"].shape[2], frame["data"].shape[1]
    logger.info(
        "Calibrating camera using:", image_width=image_width, image_height=image_height
    )
    if image_height < 100 or image_width < 100:
        logger.warn("Image dimensions look wrong! Check it out.")

    (
        normalised_named_points,
        valid_clip_ids,
    ) = get_normalized_calibration_image_points_and_clip_ids(artifacts.dataset)
    calibration_image_points = denormalize_as_named_points(
        normalised_named_points=normalised_named_points,
        image_width=image_width,
        image_height=image_height,
    )

    calibration_correspondences = get_planar_point_correspondences(
        image_points=calibration_image_points,
        world_points=corners_world_3d.copy(),
        minimal_set_count=4,
    )

    pose_correspondences = get_planar_point_correspondences(
        image_points=calibration_image_points,
        world_points=corners_world_3d.copy(),
        minimal_set_count=6,
    )

    all_world_points, all_labels = dict_to_points(corners_world_3d.copy())
    all_image_points, _ = dict_to_points(calibration_image_points.copy())
    logger.info("Calibrating camera...")

    camera_info = find_optimal_calibration_and_pose(
        valid_clip_ids=valid_clip_ids,
        calibration_correspondences=calibration_correspondences,
        pose_correspondences=pose_correspondences,
        image_height=image_height,
        image_width=image_width,
        all_image_points=all_image_points,
        all_world_points=all_world_points,
    )
    logger.info("Calibrated camera", camera_info=camera_info)
    artifacts.camera_info = camera_info
    artifacts.camera_info.save(artifacts.camera_info_path)
    return artifacts


if __name__ == "__main__":
    ANNOTATION_PATH = Path(
        "/Users/benjamindecharmoy/projects/courtvision/datasets/clip_segmentations"
    )
    ANNOTATION_DATA_PATH = Path(
        "/Users/benjamindecharmoy/projects/courtvision/datasets/clip_segmentations/data"
    )
    ANNOTATION_DATA_PATH.mkdir(exist_ok=True, parents=True)

    court_mesh_path = Path(
        "/Users/benjamindecharmoy/projects/courtvision/blender/basic_image.glb"
    )

    annotations_file = get_latest_file(ANNOTATION_PATH, "json")
    with open(annotations_file, "r") as f:
        dataset = PadelDataset(samples=json.load(f))

    artifacts = CourtVisionArtifacts(
        local_cache_path=ANNOTATION_DATA_PATH / "cache",
        dataset=dataset,
        ball_detector=BallDetector(
            model_file_or_dir=Path(
                "/Users/benjamindecharmoy/projects/courtvision/models/ball_detector/fasterrcnn_resnet50_fpn_project-1-at-2023-05-23-14-38-c467b6ad-67.pt"
            ),
            cache_dir=ANNOTATION_DATA_PATH / "cache",
        ),
        player_detector=PlayerDetector(
            model_dir=Path(
                "/Users/benjamindecharmoy/projects/courtvision/models/player_detection"
            ),
            cache_dir=ANNOTATION_DATA_PATH / "cache",
        ),
        # camera_info=None,
        camera_info_path=ANNOTATION_DATA_PATH / "cache" / "camera_info.npz",
        court_layout=PadelCourt(),
    )
    breakpoint()
    artifacts = calibrate_camera(artifacts, verbose=True)

    rr.init(
        "courtvision",
        spawn=False,
    )
    ip, port = "127.0.0.1", "9876"
    rr.connect(f"{ip}:{port}")
    current_uid = None
    for i, (frame, uid) in enumerate(
        frames_from_clip_segments(
            artifacts.dataset,
            local_path=artifacts.local_cache_path,
            stream_type=StreamType.VIDEO,
        )
    ):

        rr.set_time_sequence(uid, i)
        if uid != current_uid:
            log_court_layout(
                camera_matrix=artifacts.camera_info.camera_matrix,
                image_width=artifacts.camera_info.image_width,
                image_height=artifacts.camera_info.image_height,
                translation_vector=artifacts.camera_info.translation_vector,
                rotation_vector=artifacts.camera_info.rotation_vector,
                court_mesh_path=court_mesh_path,
            )
            current_uid = uid
        rr.log_image(
            "world/camera/image",
            frame["data"].permute(1, 2, 0).numpy().astype(np.uint8),
        )
        # Detect and log ball detections
        ball_detections = artifacts.ball_detector.predict(
            frame["data"].unsqueeze(0).float() / 255.0,
            frame_idx=i,
            clip_uid=uid,
        )
        log_ball_detections(
            detections=ball_detections,
            clip_uid=uid,
        )

        # Detect and log player detections
        player_detections = artifacts.player_detector.predict(
            frame["data"].unsqueeze(0),
            frame_idx=i,
            clip_uid=uid,
        )
        log_player_detections(
            detections=player_detections,
            clip_uid=uid,
            camera_matrix=artifacts.camera_info.camera_matrix,
            translation_vector=artifacts.camera_info.translation_vector,
            rotation_vector=artifacts.camera_info.rotation_vector,
        )
