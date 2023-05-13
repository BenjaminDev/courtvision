import random
from pathlib import Path

import torch

# import courtvision
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_choice,
    get_env,
    get_local_path,
    get_single_tag_keys,
    is_skipped,
)

# host.docker.internal -> might be better
from courtvision.data import GeneralResult, RectValue
from courtvision.models import get_fasterrcnn_ball_detection_model
from courtvision.swiss import get_latest_file
from courtvision.vis import load_timg


class DummyModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(DummyModel, self).__init__(**kwargs)

        # pre-initialize your variables here
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = schema["labels"]
        model_name = (
            "fasterrcnn_resnet50_fpn_project-1-at-2023-05-05-19-29-73700012.ptt"
        )
        self.model = get_fasterrcnn_ball_detection_model(
            model_path=get_latest_file(
                Path(
                    "/Users/benjamindecharmoy/projects/courtvision/models/ball_detector"
                ),
            )
        ).eval()

    def predict(self, tasks, **kwargs):
        """This is where inference happens:
        model returns the list of predictions based on input list of tasks

        :param tasks: Label Studio tasks in JSON format
        """
        results = []
        access_token = "45f92132fd400e8c3210e6b01f504c4de9894ebf"
        for task in tasks:
            image_path = get_local_path(
                task["data"]["image"],
                access_token=access_token,
                hostname="http://localhost:8080",
            )
            image = load_timg(image_path)
            with torch.no_grad():
                outputs = self.model(image)
            width, height = image.shape[-1], image.shape[-2]
            for output in outputs:
                for ((x1, y1, x2, y2), _, score) in zip(*output.values()):
                    results.append(
                        {
                            "result": [
                                GeneralResult(
                                    original_height=height,
                                    original_width=width,
                                    type="rectanglelabels",
                                    from_name=self.from_name,
                                    to_name=self.to_name,
                                    value=RectValue(
                                        x=(x1.item() / width) * 100.0,
                                        y=(y1.item() / height) * 100.0,
                                        width=(x2.item() - x1.item()) / width * 100.0,
                                        height=(y2.item() - y1.item()) / height * 100.0,
                                        rectanglelabels=["ball"],
                                    ),
                                ).dict(by_alias=True)
                            ],
                            "score": score.item(),
                        }
                    )
        # breakpoint()
        return results

    def fit(self, completions, workdir=None, **kwargs):
        """This is where training happens: train your model given list of completions,
        then returns dict with created links and resources

        :param completions: aka annotations, the labeling results from Label Studio
        :param workdir: current working directory for ML backend
        """
        # save some training outputs to the job result
        return {"random": random.randint(1, 10)}
