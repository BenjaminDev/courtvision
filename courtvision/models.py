from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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
