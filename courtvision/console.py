import json
from pathlib import Path

import cv2
from textual.app import App, ComposeResult
from textual.widgets import DirectoryTree

from courtvision.data import PadelDataset, StreamType, frames_from_clip_segments
from courtvision.swiss import get_latest_file

# class DirectoryTreeApp(App):
#     def compose(self) -> ComposeResult:
#         yield DirectoryTree("./")


# if __name__ == "__main__":
#     app = DirectoryTreeApp()
#     app.run()


def grab_frames_from_clips(
    frame_interval: int = 6,
    max_num_frames: int = 800,
    output_dir: Path = Path(
        "/Users/benjamindecharmoy/projects/courtvision/balldataset"
    ),
):
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
    output_dir.mkdir(exist_ok=True, parents=True)
    annotations_file = get_latest_file(ANNOTATION_PATH, "json")
    with open(annotations_file, "r") as f:
        dataset = PadelDataset(samples=json.load(f))
    for i, (frame, uid) in enumerate(
        frames_from_clip_segments(
            dataset,
            local_path=ANNOTATION_DATA_PATH,
            stream_type=StreamType.VIDEO,
        )
    ):
        if i % frame_interval == 0:
            cv2.imwrite(
                (output_dir / f"{uid}_{i:04}.png").as_posix(),
                frame["data"].permute(1, 2, 0).numpy()[:, :, ::-1],
            )
        if i > max_num_frames:
            break
