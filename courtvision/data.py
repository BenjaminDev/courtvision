from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field


class AnnotationDataPath(BaseModel):
    image: Path


class RectValue(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rectanglelabels: list[str]


class KeypointValue(BaseModel):
    x: float
    y: float
    width: float
    # height:float|None
    keypointlabels: list[str]


class GeneralResult(BaseModel):
    kind: str = Field(..., alias="type")
    original_width: int
    original_height: int
    value: Union[
        RectValue,
        KeypointValue,
    ]

    from_name: str


class Annotation(BaseModel):
    unique_id: str
    result: list[GeneralResult]


class CourtAnnotatedSample(BaseModel):
    idx: int = Field(..., alias="id")
    data: AnnotationDataPath
    annotations: list[Annotation]


class PadelDataset(BaseModel):
    samples: list[CourtAnnotatedSample]


from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import VisionDataset

from courtvision.data import Annotation, CourtAnnotatedSample, KeypointValue, RectValue
from courtvision.vis import draw_rect, load_timg


class CourtVisionDataset(VisionDataset):
    def __init__(
        self,
        dataset: PadelDataset,
        root: str,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.root = root  # TODO: See what base class does and if we can use it
        self.dataset = dataset
        super().__init__(root, transforms, transform, target_transform)

    def __len__(self):
        return len(self.dataset.samples)

    def __getitem__(self, idx) -> tuple[CourtAnnotatedSample, torch.Tensor]:
        sample = self.dataset.samples[idx]
        image = load_timg(CourtVisionDataset.find_image_path(self.root, sample=sample))
        return (
            sample,
            image,
        )

    @staticmethod
    def find_image_path(root: Path | str, sample: CourtAnnotatedSample):
        """Finds the image path from a sample"""
        root = Path(root)
        dir_name, _, frame_idx = sample.data.image.stem.split("-")[-1].partition(
            "_frame"
        )
        return root / dir_name / f"{dir_name}_frame{frame_idx}.png"

    @staticmethod
    def show_sample(annotation: list[Annotation], image: torch.Tensor):
        """Plots an image and its annotation"""

        def draw_annotaion(annotation: Annotation, image: torch.Tensor):
            bboxes = [
                r.value for r in annotation.result if isinstance(r.value, RectValue)
            ]

            original_sizes = [
                (r.original_width, r.original_height)
                for r in annotation.result
                if isinstance(r.value, RectValue)
            ]
            if bboxes:
                rects = torch.stack(
                    [
                        torch.tensor(
                            [
                                (bbox.x / 100.0) * w_h[0],
                                (bbox.y / 100.0) * w_h[1],
                                (bbox.x + bbox.width) / 100.0 * w_h[0],
                                (bbox.y + bbox.height) / 100.0 * w_h[1],
                            ]
                        ).unsqueeze(0)
                        for bbox, w_h in zip(bboxes, original_sizes)
                    ]
                ).permute(1, 0, 2)
                print(rects.shape)
                draw_rect(image, bboxes=rects)

            keypoints = [
                r.value for r in annotation.result if isinstance(r.value, KeypointValue)
            ]
            original_sizes = [
                (r.original_width, r.original_height)
                for r in annotation.result
                if isinstance(r.value, KeypointValue)
            ]
            if keypoints:
                point_width = 1.0
                rects = torch.stack(
                    [
                        torch.tensor(
                            [
                                (point.x / 100.0) * w_h[0],
                                (point.y / 100.0) * w_h[1],
                                (point.x + point_width) / 100.0 * w_h[0],
                                (point.y + point_width) / 100.0 * w_h[1],
                            ]
                        ).unsqueeze(0)
                        for point, w_h in zip(keypoints, original_sizes)
                    ]
                ).permute(1, 0, 2)

                draw_rect(image, bboxes=rects)

        for annot in annotation:
            draw_annotaion(annot, image)
        plt.imshow(image.squeeze(0).permute(1, 2, 0))
