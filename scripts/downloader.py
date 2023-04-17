import re
import string
from pathlib import Path

from pytube import YouTube


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.sub(r"", text)


data_dir = Path("data")


def normalize_clip_title(title: str):
    ignore_symbols = string.punctuation.replace("-", "")
    return remove_emojis(
        f"{title.replace(' ', '-').translate(str.maketrans('','',ignore_symbols))}_cc.mp4"
    )


# link of the video to be downloaded
# link="https://www.youtube.com/watch?v=j8U4yTXVUYI"
# link="https://www.youtube.com/watch?v=4Sfj5uyV8X4"
# link="https://www.youtube.com/watch?v=Dv-h5bxXlBY"
# link="https://www.youtube.com/watch?v=rOpI-KRaznQ"
# link="https://www.youtube.com/watch?v=Yq5Prva5C5A"
# link="https://www.youtube.com/watch?v=Mm4zEMdG81s"
# Creative Commons
# link="https://www.youtube.com/watch?v=RJ3KZCS7oAk"

# Clips
link = "https://www.youtube.com/shorts/S5igtIDi0v4"

yt = YouTube(link)
clip = yt.streams.get_highest_resolution()
print(normalize_clip_title(clip.title))
clip.download(output_path=data_dir / "raw", filename=normalize_clip_title(clip.title))
