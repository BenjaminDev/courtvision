from pathlib import Path
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader


class AnnotationDataPath(BaseModel):
    video_url: Optional[Path]
    video_local_path: Optional[Path]
    image: Optional[Path] = Field(None, alias="img")

    class Config:
        allow_population_by_field_name = True


class LabelValue(BaseModel):
    start: float
    end: float
    labels: list[str]


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
    keypointlabels: list[str]


class VideoRectSequence(BaseModel):

    frame: int
    enabled: bool
    rotation: float
    x: float
    y: float
    width: float
    height: float
    time: float


class VideoRectValue(BaseModel):
    framesCount: int
    duration: float
    sequence: list[VideoRectSequence]
    labels: list[str]
    # videorectangle: list[str]


class PolygonValue(BaseModel):
    points: list[tuple[float, float]]
    polygonlabels: list[str]


class ClipSegmentResult(BaseModel):
    original_length: Optional[float]
    clip_id: str = Field(..., alias="id")
    kind: str = Field(..., alias="type")
    value: Union[
        VideoRectValue,
        LabelValue,
    ]


class GeneralResult(BaseModel):
    kind: str = Field(..., alias="type")
    original_width: int
    original_height: int
    value: Union[
        RectValue,
        KeypointValue,
        PolygonValue,
    ]
    to_name: str = ""
    from_name: str


class Annotation(BaseModel):
    unique_id: str
    result: Union[
        list[ClipSegmentResult],
        list[GeneralResult],
    ]


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
    def collate_fn(batch):
        """Collate function for the dataloader"""
        samples, images = zip(*batch)
        return samples, torch.stack(images)

    @staticmethod
    def find_image_path(root: Path | str, sample: CourtAnnotatedSample):
        """Finds the image path from a sample"""
        #         root = Path(root)
        #         dir_name, _, frame_idx = sample.data.image.stem.partition("_frame")
        #         dir_name = dir_name.split("-", 1)[-1]
        #         filename = root / dir_name / f"{dir_name}_frame{frame_idx}.png"
        # # /Users/benjamindecharmoy/projects/courtvision/labelstudiodata/media/upload/1/b6f5d028-Highlights-TOLITO-AGUIRRE---TITO-ALLEMANDI-vs-CHIOSTRI---MELGRATTI--Sa_Lwr6ooR.png
        #         if not filename.exists():
        #             print(f"{filename=} \n{root=}!")
        #             print(f"{sample.data.image=}")
        #             raise Exception("Could not find image")
        #         return filename
        server_file_path = Path(*sample.data.image.parts[2:])  # remove /data/
        filename = Path(f"{root}/{server_file_path}")
        # print(f"{filename=}")
        return filename

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


def annotations_to_bbox(annotations: list[Annotation]):
    bboxes = []
    original_sizes = []
    for annotation in annotations:
        bboxes.extend(
            [r.value for r in annotation.result if isinstance(r.value, RectValue)]
        )
        original_sizes.extend(
            [
                (r.original_width, r.original_height)
                for r in annotation.result
                if isinstance(r.value, RectValue)
            ]
        )
    # if not bboxes:
    # raise ValueError("No bounding boxes in annotation")

    return torch.stack(
        [
            torch.tensor(
                [
                    (bbox.x / 100.0) * w_h[0],
                    (bbox.y / 100.0) * w_h[1],
                    (bbox.x + bbox.width) / 100.0 * w_h[0],
                    (bbox.y + bbox.height) / 100.0 * w_h[1],
                ]
            )
            for bbox, w_h in zip(bboxes, original_sizes)
        ]
    )


def collate_fn(batch):
    """Collate function for the dataloader"""
    samples, images = zip(*batch)
    targets = [
        {
            "boxes": annotations_to_bbox(sample.annotations),
            "labels": torch.ones(1, dtype=torch.int64),
        }
        for sample in samples
    ]

    return targets, [o.squeeze(0) for o in images]


def validate_dataloader(dataloader: DataLoader):
    for (targets, images) in dataloader:
        assert all(o["boxes"].shape for o in targets)
        assert all(o.shape for o in images)


def get_keypoints_as_dict(
    results: list[GeneralResult],
) -> dict[str, tuple[float, float]]:
    """Go through the results and return a dict of keypoints

    Args:
        results (list[GeneralResult]): List of results from the annotation

    Returns:
        dict[str, tuple[float, float]]: keypoints in absolute coordinates eg: keypoints["{some keypoint}"] = (x, y)
    """
    keypoints = {}
    for result in results:
        if isinstance(result.value, KeypointValue):
            keypoints[result.value.keypointlabels[0]] = (
                result.value.x / 100.0 * result.original_width,
                result.value.y / 100.0 * result.original_height,
            )
    return keypoints


def dict_to_points(
    keypoints: dict[str, tuple[float, float]]
) -> tuple[np.array, list[str]]:
    """Unpacks a dict of keypoints into a np.array of points and a list of labels

    Args:
        keypoints (dict[str, tuple[float, float]]): Dict of keypoints

    Returns:
        np.array, list[str]: Nx2 array of points and list of labels
    """
    keypoints = dict(sorted(keypoints.items(), key=lambda x: x[0]))
    return np.array(list(keypoints.values())).astype(np.float32), list(keypoints.keys())


def download_data_item(s3_uri: str, local_path: Path, s3_client=None, use_cached=True):

    if use_cached and local_path.exists():
        return local_path

    if s3_client is None:
        import boto3

        session = boto3.Session(profile_name="courtvision-padel-dataset")
        s3_client = session.client("s3", region_name="us-east-1")
    bucket_name = s3_uri.parents[-3].name
    object_name = "/".join(s3_uri.parts[-3:])
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with open(local_path, "wb") as fp:
        s3_client.download_fileobj(bucket_name, object_name, fp)
    return local_path


import enum
from hashlib import md5

import torchvision


class StreamType(enum.Enum):
    VIDEO = "video"
    AUDIO = "image"


def frames_from_clip_segments(
    dataset: PadelDataset, local_path: Path, stream_type: StreamType = StreamType.VIDEO
) -> tuple[dict[str, torch.Tensor], str]:
    """
    Graps frames for each clip segment in the dataset. A unique id is generated for each clip segment.
    Frames can be either audio or video frames.

    Args:
        dataset (PadelDataset): A dataset of annotated clips
        local_path (Path): if the file is not already downloaded, it will be downloaded to this path
        stream_type (StreamType, optional): Either `StreamType.VIDEO` or `StreamType.AUDIO`. Defaults to StreamType.VIDEO.

    Returns:
        tuple[dict[str, torch.Tensor], str]: returns `{"data": torch.Tensor, "pts": torch.Tensor}, unique_id`

        where `unique_id` is the md5 of the annotation unique_id and the start and end times of the clip.
        And `pts` is a presentation timestamp of the frame expressed in seconds.

    Yields:
        Iterator[tuple[dict[str, torch.Tensor], str]]: yeilds `{"data": torch.Tensor, "pts": torch.Tensor}, unique_id`
    """
    for sample in dataset.samples:
        sample.data.video_local_path = download_data_item(
            s3_uri=sample.data.video_url,
            local_path=local_path
            / sample.data.video_url.parent.name
            / sample.data.video_url.name,
        )
    for sample in dataset.samples:
        for annotation in sample.annotations:
            for result in annotation.result:
                if isinstance(result, ClipSegmentResult):
                    start_time = result.value.start
                    end_time = result.value.end
                    reader = torchvision.io.VideoReader(
                        sample.data.video_local_path.as_posix(), stream_type.value
                    )
                    reader.seek(start_time)
                    while frame := next(reader):
                        if frame["pts"] < start_time:
                            # seeks is not always accuarte!
                            # burn frames until we get to the right time.
                            # Alternative - build torchvison from source with video_reader backend
                            continue
                        if frame["pts"] > end_time:
                            break
                        yield frame, md5(
                            f"{start_time}{end_time}{annotation.unique_id}".encode()
                        ).hexdigest()


def get_normalized_calibration_image_points_and_clip_ids(
    dataset: PadelDataset,
) -> tuple[dict[str, tuple[float, float]], set[str]]:
    """
    Note: This assumes that the calibration points are the only annotations with a VideoRectValue
    and the points of the same label are in the same place as the last one which will be used.

    Note: Points are normalized to 0-1. Not -1 to 1 like in kornia.
    Args:
        dataset (PadelDataset): Dataset descibing a video with calibration points.

    Returns:
        image_points (dict[str, tuple[float, float]]): Returns a dict of image points in normalized coordinates. And
        the clip_ids (set[str]) that are accociated with the calibration points.
    """
    calibration_image_points = {}
    clip_source = set([])
    for sample in dataset.samples:
        for annotation in sample.annotations:
            if annotation.data.video_url:
                clip_source.add(annotation.data.video_url)
            elif annotation.data.video_local_path:
                clip_source.add(annotation.data.video_local_path)
            elif annotation.data.image:
                clip_source.add(annotation.data.image)
            else:
                raise ValueError("No clip source found")

            for result in annotation.result:
                if isinstance(result.value, VideoRectValue):
                    for label, rect in zip(result.value.labels, result.value.sequence):
                        calibration_image_points[label] = (
                            (rect.x + rect.width / 2) / 100.0,
                            (rect.y + rect.height / 2) / 100.0,
                        )
    return calibration_image_points, clip_source
