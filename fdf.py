"""Flickr Diverse Faces"""

import os
import urllib
import datasets
import pandas as pd

_CITATION = """\
@InProceedings{10.1007/978-3-030-33720-9_44,
author="Hukkel{\aa}s, H{\aa}konand Mester, Rudolf
and Lindseth, Frank",
title="DeepPrivacy: A Generative Adversarial Network for Face Anonymization",
booktitle="Advances in Visual Computing",
year="2019",
publisher="Springer International Publishing",
pages="565--578",
isbn="978-3-030-33720-9"}
"""

_DESCRIPTION = """\
Flickr Diverse Faces (FDF) is a dataset with 1.5M faces "in the wild".
FDF has a large diversity in terms of facial pose, age, ethnicity, occluding objects, facial painting, and image background.
The dataset is designed for generative models for face anonymization, and it was released with the paper DeepPrivacy: A Generative Adversarial Network for Face Anonymization.

The dataset was crawled from the website Flickr (YFCC-100M dataset) and automatically annotated.
Each face is annotated with 7 facial landmarks (left/right ear, lef/right eye, left/right shoulder, and nose), and a bounding box of the face.
Our paper goes into more detail about the automatic annotation.
"""

_LICENSE = """\
https://www.apache.org/licenses/LICENSE-2.0
"""

_URL = (
    "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/"
)
RESOURCES = {
    "metadata": urllib.parse.urljoin(
        _URL,
        "b704049a-d465-4a07-9cb3-ca270ffab80292e4d5ac-6172-4d37-bf63-4438f61f8aa0e1f6483d-5d45-40b5-b356-10b71fc00e89",
    ),
    "cc-by-nc-sa-2": urllib.parse.urljoin(
        _URL,
        "33bb6132-a30e-4169-a09e-48b94cd5e09010fd8f59-1db9-4192-96f0-83d18adf50b4ee3ab25c-b8f8-4c40-a8ba-eaa64d830412",
    ),
    "cc-by-2": urllib.parse.urljoin(
        _URL,
        "cb545564-120f-4f35-8b68-63e59e4fd273b1c36452-21e7-4976-85dd-a86c0738ebc256264f20-f969-4a82-ae1c-f6345d8e8d1f",
    ),
    "cc-by-sa-2": urllib.parse.urljoin(
        _URL,
        "4e5c27bd-f5fd-4dd3-bf2b-4434a8952df0ff4d11f8-e993-4517-9378-b35d89e7882ecae76ce7-88a0-417b-b5d8-e030863e97f6",
    ),
    "cc-by-nc-2": urllib.parse.urljoin(
        _URL,
        "da46d666-4378-4e75-9182-e683ebe08f2e9203750f-7ed8-42f7-8bce-90a7dfddd3764401df36-a3c3-4d39-ae8e-0cb4397e5c74",
    ),
}


class FlickrDiverseFaces(datasets.GeneratorBasedBuilder):
    """FlickrDiverseFaces (FDF) Builder class."""

    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="fdf256",
            version=VERSION,
            description="The Flickr Diverse Faces 256 (FDF256)",
        ),
        # datasets.BuilderConfig(name="cc-by-nc-sa-2", version=VERSION, description="All images regardless of copyright."),
        # datasets.BuilderConfig(name="cc-by-2", version=VERSION, description="All images regardless of copyright."),
        # datasets.BuilderConfig(name="cc-by-nc-2", version=VERSION, description="All images regardless of copyright."),
    ]

    DEFAULT_CONFIG_NAME = "fdf256"

    def _info(self):
        if self.config.name == "fdf256":
            _CITATION = """\
                @inproceedings{hukkelas23DP2,
                author={Hukkelås, Håkon and Lindseth, Frank},
                booktitle={2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
                title={DeepPrivacy2: Towards Realistic Full-Body Anonymization}, 
                year={2023},
                volume={},
                number={},
                pages={1329-1338},
                doi={10.1109/WACV56688.2023.00138}}
            """
            features = datasets.features.Features(
                {
                    "image": datasets.features.Image(),
                    "landmarks": {
                        "nose": {
                            "x": datasets.features.Value("float32"),
                            "y": datasets.features.Value("float32"),
                        },
                        "r_eye": {
                            "x": datasets.features.Value("float32"),
                            "y": datasets.features.Value("float32"),
                        },
                        "l_eye": {
                            "x": datasets.features.Value("float32"),
                            "y": datasets.features.Value("float32"),
                        },
                        "r_ear": {
                            "x": datasets.features.Value("float32"),
                            "y": datasets.features.Value("float32"),
                        },
                        "l_ear": {
                            "x": datasets.features.Value("float32"),
                            "y": datasets.features.Value("float32"),
                        },
                        "r_shoulder": {
                            "x": datasets.features.Value("float32"),
                            "y": datasets.features.Value("float32"),
                        },
                        "l_shoulder": {
                            "x": datasets.features.Value("float32"),
                            "y": datasets.features.Value("float32"),
                        },
                    },
                    "bbox": {
                        "x_min": datasets.features.Value("uint16"),
                        "y_min": datasets.features.Value("uint16"),
                        "x_max": datasets.features.Value("uint16"),
                        "y_max": datasets.features.Value("uint16"),
                    },
                    "metadata": {
                        "license": datasets.features.Value("string"),
                        "photo_url": datasets.features.Value("string"),
                    },
                }
            )
        else:
            raise Exception("Config not recognized")

        return datasets.DatasetInfo(
            description=_DESCRIPTION + self.config.description,
            features=features,
            homepage="https://github.com/hukkelas/FDF/tree/master",
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        output_files: dict = dl_manager.download_and_extract(RESOURCES)

        data_dirs: dict = {"train": [], "val": []}

        metadata: str = output_files.pop("metadata")

        if self.config.name == "fdf256":
            for directory in output_files.values():
                data_dirs["train"].append(os.path.join(directory, "train", "images"))
                data_dirs["val"].append(os.path.join(directory, "val", "images"))
        else:
            raise ValueError("No config named %s" % (self.config.name))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dirs["train"],
                    "metadata": os.path.join(metadata, "train"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_dir": data_dirs["val"],
                    "metadata": os.path.join(metadata, "val"),
                },
            ),
        ]

    def _generate_examples(self, data_dir: list, metadata: str):
        paths: list = self.get_image_paths(data_dir=data_dir)
        image_list = [int(os.path.basename(x.removesuffix(".png"))) for x in paths]

        metadata: pd.DataFrame = pd.read_json(
            os.path.join(metadata, "fdf_metainfo.json"),
            convert_dates=True,
            orient="index",
        ).loc[image_list][["photo_url", "license", "bounding_box", "landmark"]]

        bounds = self.load_bounds(bounds=metadata.pop("bounding_box"))

        landmarks = self.load_landmarks(landmarks=metadata.pop("landmark"))

        metadata = metadata.to_dict("records")

        for path, bound, landmark, meta in zip(paths, bounds, landmarks, metadata):
            key = "%s/%s/%s/%s" % (
                os.path.basename(path.removesuffix(".png")),
                bound,
                landmark,
                meta,
            )
            # example = {"image": path, "landmarks": landmark, "bbox": bound, "metadata": meta}
            # print(example)
            yield (
                key,
                {"image": path, "landmarks": landmark, "bbox": bound, "metadata": meta},
            )

    def get_image_paths(self, data_dir: list) -> list:
        paths: list = []
        for ddir in data_dir:
            paths += [os.path.join(data_dir, ddir, x) for x in os.listdir(ddir)]
        paths = sorted(
            paths, key=lambda x: int(os.path.basename(x).removesuffix(".png"))
        )
        return paths

    def load_bounds(self, bounds: pd.Series) -> list:
        df = pd.DataFrame(
            bounds.to_list(), columns=["x_min", "y_min", "x_max", "y_max"]
        )
        return df.to_dict("records")

    def load_landmarks(self, landmarks: pd.Series) -> dict:
        landmarks = [
            {
                "nose": {"x": landmark[0], "y": landmark[1]},
                "r_eye": {"x": landmark[2], "y": landmark[3]},
                "l_eye": {"x": landmark[4], "y": landmark[5]},
                "r_ear": {"x": landmark[6], "y": landmark[7]},
                "l_ear": {"x": landmark[8], "y": landmark[9]},
                "r_shoulder": {"x": landmark[10], "y": landmark[11]},
                "l_shoulder": {"x": landmark[12], "y": landmark[13]},
            }
            for landmark in landmarks.to_list()
        ]
        return landmarks
