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
