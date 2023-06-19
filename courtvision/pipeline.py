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
            model_dir=Path(
                "/Users/benjamindecharmoy/projects/courtvision/models/ball_detection"
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
        ball_detections = artifacts.ball_detector.predict(
            frame["data"].unsqueeze(0).float(),
            frame_idx=i,
            clip_uid=uid,
        )
        if len(ball_detections[0]["boxes"]) > 0:
            breakpoint()
