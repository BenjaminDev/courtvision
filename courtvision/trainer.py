import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms
from torchvision.datasets import MNIST

import wandb
from courtvision.models import (
    BallDetector,
    PlayerDetector,
    get_fasterrcnn_ball_detection_model,
)
from courtvision.swiss import get_latest_file
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from courtvision.data import validate_dataloader    


class BallDetectorModel(pl.LightningModule):
    def __init__(self, model_path: Path = None):
        super().__init__()
        if model_path is not None:

            self.model = get_fasterrcnn_ball_detection_model(
                model_path=model_path,
            )
            self.model.train()

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss = losses.item()
        self.log("train_loss", loss)
        return losses

    def validation_step(self, val_batch, batch_idx):

        images, targets = val_batch
        self.train()  # set to train mode to get losses. TODO: fix this see `on_validation_model_train` ?
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss = losses.item()
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        
        images, targets = next(iter(self.trainer.val_dataloaders))
        images = [img.to("cuda") for img in images]
        self.eval()  # set to eval mode to get predictions
        with torch.no_grad():
            preds = self.model(images)
        metric = MeanAveragePrecision(iou_type="bbox")

        # Update metric with predictions and respective ground truth
        preds_cpu = [{k: v.cpu() for k, v in pred.items()} for pred in preds]
        metric.update(preds_cpu, targets)

        # Compute the results
        result = metric.compute()
        for k, v in result.items():
            self.log(f"val_{k}", v)

        log_wb_image_and_bbox(
            images=images,
            preds=preds,
            targets=targets,
            logger=self.trainer.logger.experiment,
            global_step=self.global_step,
        )
          


# from typing import Any
# def get_wandb_logger(loggers:list[Any])->pl.loggers.WandbLogger:
#     for logger in loggers:
#         if isinstance(logger, pl.loggers.WandbLogger):
#             return logger
#     raise ValueError("No WandbLogger found in loggers")


def log_wb_image_and_bbox(
    images: torch.tensor,
    preds: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    logger: wandb.sdk.wandb_run.Run,
    global_step: int,
):
    images_to_log = []
    print(f"\n\nLogging images {global_step}\n\n")
    if len(preds) < 1:
        breakpoint()
    for image, pred, target in zip(images, preds, targets):
        image_height, image_width = image.shape[-2:]
        images_to_log.append(
            wandb.Image(
                image.permute(1, 2, 0).cpu().numpy(),
                boxes={
                    "predictions": {
                        "box_data": [
                            {
                                "position": {
                                    "minX": float(x_min) / image_width,
                                    "maxX": float(x_max) / image_width,
                                    "minY": float(y_min) / image_height,
                                    "maxY": float(y_max) / image_height,
                                },
                                "class_id": 1,
                                "box_caption": "ball",
                                "scores": {"score": float(score)},
                            }
                            for (x_min, y_min, x_max, y_max), score in zip(
                                pred["boxes"].cpu().numpy(),
                                pred["scores"].cpu().numpy(),
                                strict=True,
                            )
                        ]
                    },
                    "targets": {
                        "box_data": [
                            {
                                "position": {
                                    "minX": float(x_min) / image_width,
                                    "maxX": float(x_max) / image_width,
                                    "minY": float(y_min) / image_height,
                                    "maxY": float(y_max) / image_height,
                                },
                                "class_id": 1,
                                "box_caption": "ball",
                            }
                            for x_min, y_min, x_max, y_max in target["boxes"]
                            .cpu()
                            .numpy()
                        ],
                    },
                },
            )
        )

    logger.log({"image": images_to_log})


from pytorch_lightning.callbacks import ModelCheckpoint

from courtvision.data import CourtVisionBallDataset, CourtVisionDataset, PadelDataset

if __name__ == "__main__":
    from courtvision.config import CourtVisionSettings

    settings = CourtVisionSettings()
    ANNOTATION_PATH = Path(settings.datasets_path / "ball_dataset")
    ANNOTATION_DATA_PATH = ANNOTATION_PATH / "data"
    ANNOTATION_DATA_PATH.mkdir(exist_ok=True, parents=True)

    annotations_file = get_latest_file(ANNOTATION_PATH, "json")
    with open(annotations_file, "r") as f:
        padel_dataset = PadelDataset(
            samples=json.load(f), local_data_dir=ANNOTATION_DATA_PATH
        )

    courtvision_dataset = CourtVisionBallDataset(
        dataset=padel_dataset,
        root=ANNOTATION_DATA_PATH,
        download=True,
    )
    from pytorch_lightning.loggers import WandbLogger

    wandb_logger = WandbLogger(
        project=settings.wb_project,
        save_dir=settings.wb_save_dir,
    )
    
    ball_dataset_train, balldataset_val = random_split(courtvision_dataset, [len(courtvision_dataset)-20, 20 ])

    train_loader = DataLoader(
        ball_dataset_train, batch_size=1, collate_fn=CourtVisionBallDataset.collate_fn
    )
    validate_dataloader(train_loader)
    val_loader = DataLoader(
        balldataset_val, batch_size=1, collate_fn=CourtVisionBallDataset.collate_fn
    )
    validate_dataloader(val_loader)

    # model
    model = BallDetectorModel(
        model_path=settings.ball_models_dir / settings.ball_model_name
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # training

    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.ball_checkpoints_dir, save_top_k=5, monitor="val_loss"
    )

    checkpoint_weights_only = ModelCheckpoint(
        dirpath=settings.ball_checkpoints_weights_only, save_top_k=5, monitor="val_loss"
    )

    trainer = pl.Trainer(
        limit_train_batches=0.5,
        accelerator=device,
        log_every_n_steps=1,
        logger=wandb_logger,
        num_sanity_val_steps=-1,
        callbacks=[checkpoint_weights_only]
    )
    trainer.fit(model, train_loader, val_loader)
