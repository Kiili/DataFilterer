import math

import PIL.Image
import numpy as np
import torch
import os

from filterable_interface import FilterableInterface


class Coco128Dataset(torch.utils.data.Dataset):
    def __init__(self, batch_size=1):
        self.path = 'data'

        if not os.path.exists(self.path + "/coco128"):  # download and extract coco128 dataset
            cmd = ["curl", "-L", "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip", "-o",
                   "coco128.zip", "-#", "&&", "unzip", "-q", "coco128.zip", "-d", self.path, "&&", "rm", "coco128.zip"]
            os.system(" ".join(cmd))

        self.batch_size = batch_size
        prefix = self.path + "/coco128/images/train2017/"
        self.image_files = [prefix + file for file in os.listdir(prefix)]

    def __len__(self):
        return math.ceil(len(self.image_files) / self.batch_size)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        imgs = []
        paths = []
        for i in range(self.batch_size * idx, min(len(self.image_files), self.batch_size * (idx + 1))):
            paths.append(self.image_files[i])
            imgs.append(PIL.Image.open(self.image_files[i]).resize(size=(640, 640)))

        # Return batch of images and file_paths as metadata
        return imgs, paths


class Yolov5Example(FilterableInterface):

    def __init__(self):
        # yolov5 pretrained on coco
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # first 128 images of coco
        self.filterable_dataset = Coco128Dataset(batch_size=5)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_filterable_dataset(self):
        return self.filterable_dataset

    def get_image(self, arg) -> np.ndarray:
        img = PIL.Image.open(arg).resize(size=(640, 640))
        return np.array(img)


if __name__ == '__main__':
    # example usage
    from data_filterer import DataFilterer

    filterer_instance = DataFilterer(model=Yolov5Example(),
                                     layer="model.model.model.10",  # see print(model) for layer names
                                     device="cuda")

    idxs = filterer_instance.get_idxs(semantic_percentage=0.09,
                                      outlier_percentage=0.01,
                                      downscale_dim=None  # no dimensionality reduction to maximize accuracy
                                      )
    # :idxs: images that are in the filtered dataset

    similars, outliers = filterer_instance.get_filtered_out()
    # :similars: dict(k, v) where v images were filtered out and considered too similar to k
    # :outliers: images that were considered as outliers

    # visualize the dataset
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.use("GTK3Agg")
    filterer_instance.get_fig(plt)
    plt.show()
