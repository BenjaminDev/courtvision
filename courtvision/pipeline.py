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
import ultralytics
from kornia.geometry import unproject_points
from structlog import wrap_logger
from torch.functional import F

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
    calibrate_camera,
    convert_obj_points_to_planar,
    corners_world_3d,
    denormalize_as_named_points,
    find_optimal_calibration_and_pose,
    get_planar_point_correspondences,
    get_planar_points_padel_court,
    project_points_to_base_plane,
)
from courtvision.models import BallDetector, PlayerDetector
from courtvision.swiss import get_latest_file, mark_as_deprecated
from courtvision.trackers import ParticleFilter
from courtvision.vis import (
    colours_per_player_idx,
    log_ball_detections,
    log_court_layout,
    log_player_detections,
)

logger = structlog.get_logger()


class Verbosity(Enum):
    SILENT = 0
    PROGRESS = 1
    DEBUG = 2


def pipeline(
    artifacts: CourtVisionArtifacts,
):
    # Calibrate camera
    artifacts = calibrate_camera(artifacts, logger=logger)


if __name__ == "__main__":
    # TODO: Make this a proper CLI using Typer. https://github.com/BenjaminDev/courtvision/issues/4
    ANNOTATION_PATH = Path("../datasets/clip_segmentations")
    ANNOTATION_DATA_PATH = Path("../datasets/clip_segmentations/data")
    ANNOTATION_DATA_PATH.mkdir(exist_ok=True, parents=True)

    court_mesh_path = Path("../blender/basic_image.glb")

    annotations_file = get_latest_file(ANNOTATION_PATH, "json")
    with open(annotations_file, "r") as f:
        dataset = PadelDataset(samples=json.load(f))

    artifacts = CourtVisionArtifacts(
        local_cache_path=ANNOTATION_DATA_PATH / "cache",
        dataset=dataset,
        ball_detector=BallDetector(
            model_file_or_dir=Path(
                "../models/ball_detector/fasterrcnn_resnet50_fpn_project-1-at-2023-05-23-14-38-c467b6ad-67.pt"
            ),
            cache_dir=ANNOTATION_DATA_PATH / "cache",
        ),
        ball_tracker=ParticleFilter(
            num_particles=10_000,
            court_size=torch.tensor(
                [PadelCourt.width, PadelCourt.length, PadelCourt.backwall_fence_height]
            ),
        ),
        player_detector=PlayerDetector(
            model_dir=Path("../models/player_detection"),
            cache_dir=ANNOTATION_DATA_PATH / "cache",
        ),
        camera_info_path=ANNOTATION_DATA_PATH / "cache" / "camera_info.npz",
        court_layout=PadelCourt(),
    )

    # Calibrate camera from annotations in the dataset
    artifacts = calibrate_camera(artifacts, logger=logger)
    artifacts.ball_tracker.reset(
        num_particles=10_000,
        court_size=torch.tensor(
            [PadelCourt.width, PadelCourt.length, PadelCourt.backwall_fence_height]
        ),
        world_to_cam=artifacts.camera_info.world_space_to_camera_space(),
        cam_to_image=torch.tensor(artifacts.camera_info.camera_matrix),
    )
    rr.init(
        "courtvision",
        spawn=False,
        recording_id="test",
    )
    ip, port = (
        "127.0.0.1",
        "9876",
    )  # TODO: Make this configurable https://github.com/BenjaminDev/courtvision/issues/4
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
            if current_uid is not None:
                pass
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
        # TODO: use variable dt for prediction. Use frame["pts"]
        artifacts.ball_tracker.predict(dt=1 / 30.0)
        for ball_detection in ball_detections:
            for (bx1, by1, bx2, by2), ball_score in zip(
                ball_detection["boxes"][:4], ball_detection["scores"]
            ):
                obs_state = torch.tensor(
                    [
                        (bx1 + bx2) / 2.0,
                        (by1 + by2) / 2.0,
                    ]
                ).to(dtype=torch.float32)

                artifacts.ball_tracker.update(obs_state, ball_score)
                # TODO: Move this to a separate function in vis.py
                rr.log_points(
                    "world/ball_state",
                    positions=artifacts.ball_tracker.xyz,
                )
                rr.log_point(
                    "world/tracker_mean",
                    artifacts.ball_tracker.xyz_mean,
                    color=(255, 222, 0),
                    radius=2.0,
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
