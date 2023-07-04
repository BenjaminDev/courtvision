from pathlib import Path
from typing import Any

import structlog
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

from courtvision.swiss import get_latest_file

logger = structlog.get_logger("courtvision.models")


def get_fasterrcnn_ball_detection_model(model_path: None | Path = None) -> FasterRCNN:
    """Fetches a FasterRCNN model for ball detection.
    If model_path is None, the model is pretrained on COCO.
    If model_path is a Path, the model is loaded from the path.

    Args:
        model_path (None | Path, optional): Path do model weights that will be loaded. Defaults to None.

    Returns:
        FasterRCNN: A ball detection model using FasterRCNN
    """

    pretrained = model_path is None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, pretrained=pretrained
    )
    num_classes = 2  # 1 class (ball) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


def get_yolo_player_detection_model(model_path: None | Path = None) -> Any:
    """Fetches a pretrained YOLO model for player detection.

    Args:
        model_path (None | Path, optional): Unused!. Defaults to None.

    Returns:
        Any: Yolo model
    """
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    return model


def get_yolov8_player_detection_model(model_path: None | Path = None):
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt", task="detect")

    model.classes = [0]
    model.conf = 0.6
    model.max_det = 4
    model.tracker = "courtvision/models/botsort.yml"
    return model


def get_ball_detection_model(model_path: Path) -> "BallDetectorModel":
    """Grabs a trained ball detection model from a path.

    Args:
        model_path (Path): Path to the model weights. A .ckpt file.

    Returns:
        BallDetectorModel: A trained BallDetectorModel from a checkpoint.
    """
    from courtvision.trainer import BallDetectorModel  # TODO: move to models.py

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return BallDetectorModel.load_from_checkpoint(model_path, map_location=device)


class BallDetector:
    PIPELINE_NAME = "ball_detection"

    def __init__(self, model_file_or_dir: Path, cache_dir: Path):
        if model_file_or_dir.is_dir():
            self.model_path = get_latest_file(model_file_or_dir)
        else:
            self.model_path = model_file_or_dir

        self.model = get_ball_detection_model(model_path=self.model_path)
        self.cache_dir = cache_dir
        self.model.eval()

    def predict(
        self, image: torch.Tensor, frame_idx: int, clip_uid: str
    ) -> dict[str, torch.Tensor]:
        """Predicts ball detections for a given frame.
        !!! note
            This method caches the detections on disk.
        Args:
            image (torch.Tensor): Image tensor of shape (1,3,H,W)
            frame_idx (int): frame index
            clip_uid (str): clip uid that identifies the clip uniquely.

        Returns:
            dict[str, torch.Tensor]: A dict tensor ball detections.
        """
        cache_path = (
            self.cache_dir
            / self.PIPELINE_NAME
            / clip_uid
            / f"detections_at_{frame_idx}.pt"
        )
        if not cache_path.is_dir():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            return torch.load(cache_path)
        else:
            with torch.no_grad():
                detections = self.model(image)
            torch.save(detections, cache_path)
            return detections


class PlayerDetector:
    PIPELINE_NAME = "player_detection"

    def __init__(self, model_dir: Path, cache_dir: Path):
        self.model_path = get_latest_file(model_dir)
        self.cache_dir = cache_dir
        self.model = get_yolov8_player_detection_model(model_path=self.model_path)
        # self.model.eval()

    def predict(
        self, image: torch.Tensor, frame_idx: int, clip_uid: str
    ) -> dict[str, torch.Tensor]:
        """Predicts player detections for a given frame.
        !!! note
            This method caches the detections on disk.
        Args:
            image (torch.Tensor): Image tensor of shape (1,3,H,W)
            frame_idx (int): frame index
            clip_uid (str): clip uid that identifies the clip uniquely.

        Returns:
            dict[str, torch.Tensor]: Dict of player detections.
        """
        cache_path = (
            self.cache_dir
            / self.PIPELINE_NAME
            / clip_uid
            / f"detections_at_{frame_idx}.pt"
        )
        if not cache_path.is_dir():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            return torch.load(cache_path)
        else:
            with torch.no_grad():
                detections = self.model.track(
                    source=image.squeeze(0).permute(1, 2, 0).numpy(),
                    persist=True,
                )
            torch.save(detections, cache_path)
            return detections
