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


def get_yolo_player_detection_model(model_path: None | Path = None):
    import torch

    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    return model


def get_yolov8_player_detection_model(model_path: None | Path = None):
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt", task="detect")

    model.classes = [0]
    model.conf = 0.6
    model.max_det = 4
    model.tracker = "/Users/benjamindecharmoy/projects/courtvision/botsort.yml"
    # results = model.track(
    #     source=RAW_CLIP_PATH.as_posix(),
    #     # tracker="/Users/benjamindecharmoy/projects/courtvision/bytetrack.yaml",
    #     tracker="/Users/benjamindecharmoy/projects/courtvision/botsort.yml",
    #     classes=[0],
    #     max_det=4,
    #     save=True,
    # )
    return model


class BallDetector:
    PIPELINE_NAME = "ball_detection"

    def __init__(self, model_file_or_dir: Path, cache_dir: Path):
        if model_file_or_dir.is_dir():
            self.model_path = get_latest_file(model_file_or_dir)
        else:
            self.model_path = model_file_or_dir
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
        self.model = get_yolov8_player_detection_model(model_path=self.model_path)
        # self.model.eval()

    def predict(
        self, image: torch.Tensor, frame_idx: int, clip_uid: str
    ) -> torch.Tensor:
        from torchvision.transforms.functional import resize

        resized_image = image  # resize(image, (320,640))
        cache_path = (
            self.cache_dir
            / self.PIPELINE_NAME
            / clip_uid
            / f"detections_at_{frame_idx}.pt"
        )
        if not cache_path.is_dir():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file() and False:
            return torch.load(cache_path)
        else:
            with torch.no_grad():
                detections = self.model.track(
                    source=resized_image.squeeze(0).permute(1, 2, 0).numpy(),
                    persist=True,
                )
            print(detections[0].boxes)
            torch.save(detections, cache_path)
            return detections
