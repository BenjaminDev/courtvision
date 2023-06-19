from pathlib import Path

import structlog
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from courtvision.swiss import get_latest_file

logger = structlog.get_logger("courtvision.models")


def get_fasterrcnn_ball_detection_model(model_path: None | Path = None):
    # return torch.load(model_path)
    pretrained = model_path is None
    # raise NotImplementedError()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, pretrained=False
    )
    num_classes = 2  # 1 class (ball) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model


class BallDetector:
    PIPELINE_NAME = "ball_detection"

    def __init__(self, model_dir: Path, cache_dir: Path):
        self.model_path = get_latest_file(model_dir)
        self.model = get_fasterrcnn_ball_detection_model(model_path=self.model_path)
        self.cache_dir = cache_dir
        self.model.eval()

    def predict(
        self, image: torch.Tensor, frame_idx: int, clip_uid: str
    ) -> torch.Tensor:
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
        self.model = get_fasterrcnn_ball_detection_model(model_path=self.model_path)
        self.model.eval()

    def predict(self, image: torch.Tensor, frame_idx: int) -> torch.Tensor:

        cache_path = (
            self.cache_dir / self.PIPELINE_NAME / f"detections_at_{frame_idx}.pt"
        )
        if cache_path.is_file():
            return torch.load(cache_path)
        else:
            with torch.no_grad():
                detections = self.model(image)
                if len(detections[0]["boxes"]) > 0:
                    torch.save(detections, cache_path)
                else:
                    breakpoint()
                    logger.warning("No detections", frame_idx=frame_idx)
            return detections
