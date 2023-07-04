import shlex
from pathlib import Path
from subprocess import check_output

raw_data_dir = Path(
    "/Users/benjamindecharmoy/projects/courtvision/datasets/clip_segmentations/data/cache/0000"
)
overwrite = False

for video_path in raw_data_dir.glob("*.mp4"):
    print(f"Processing {video_path}")
    output_dir = Path("/Users/benjamindecharmoy/projects/courtvision/ball_images")
    if output_dir.exists() and not overwrite:
        continue
    output_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg -i input.mp4 -r 30 -f image2 output-%d.png
    frames_output = check_output(
        shlex.split(
            f"ffmpeg -i {video_path} -r 30 -f image2 {output_dir}/{video_path.stem}_frame_%04d.png"
        )
    )
    # audio_output = check_output(
    #     shlex.split(
    #         f"ffmpeg -i {video_path} -q:a 0 -map a {output_dir}/{video_path.stem}_audio.wav"
    #     )
    # )
