import re
import string
from pathlib import Path

# from pytube import YouTube
from yt_dlp import YoutubeDL


def monkey_patch():
    from typing import List

    from pytube.cipher import (
        get_throttling_function_array,
        get_throttling_plan,
        get_transform_map,
        get_transform_plan,
    )
    from pytube.exceptions import RegexMatchError

    def __init__(self, js: str):
        self.transform_plan: List[str] = get_transform_plan(js)
        # var_regex = re.compile(r"^\w+\W")
        var_regex = re.compile(r"^\$*\w+\W")
        var_match = var_regex.search(self.transform_plan[0])
        if not var_match:
            raise RegexMatchError(caller="__init__", pattern=var_regex.pattern)
        var = var_match.group(0)[:-1]
        self.transform_map = get_transform_map(js, var)
        self.js_func_patterns = [
            r"\w+\.(\w+)\(\w,(\d+)\)",
            r"\w+\[(\"\w+\")\]\(\w,(\d+)\)",
        ]

        self.throttling_plan = get_throttling_plan(js)
        self.throttling_array = get_throttling_function_array(js)

        self.calculated_n = None

    from pytube.cipher import Cipher

    Cipher.__init__ = __init__


monkey_patch()


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

#
# link = "https://www.youtube.com/watch?v=nBq3h6YPVuU"
# link = "https://www.youtube.com/watch?v=JLsMDz6Bcc8"

# Creative Commons
# link="https://www.youtube.com/watch?v=RJ3KZCS7oAk"
link = "https://www.youtube.com/watch?v=IefR37b6qaE"

# Clips
# link = "https://www.youtube.com/shorts/S5igtIDi0v4"

# Tennis
# link = "https://www.youtube.com/watch?v=2JkVp28oSbk"
URLS = [link]
# yt = YouTube(link)
# clip = yt.streams.get_highest_resolution()
# print(normalize_clip_title(clip.title))
# clip.download(output_path=data_dir / "raw", filename=normalize_clip_title(clip.title))
with YoutubeDL() as ydl:
    ydl.download(URLS)
