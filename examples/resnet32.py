
import numpy as np
import torch
import torchvision
from torchvision import transforms

from filterable_interface import FilterableInterface


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self):
        super().__init__(root='./data/cifar-10',
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])]))

    def __getitem__(self, idx):
        img, label = super().__getitem__(index=idx)

        # Return the image and the index of the image as metadata
        return img, idx


class Resnet32Example(FilterableInterface):

    def __init__(self):
        # A pretrained model on CIFAR10
        self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)

        self.filterable_dataset = torch.utils.data.DataLoader(dataset=CIFAR10Dataset(), batch_size=128)

        # dataset without transformations for viewing images
        self.dataset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_filterable_dataset(self):
        return self.filterable_dataset

    def get_image(self, arg) -> np.ndarray:
        img, label = self.dataset[arg]
        return np.asarray(img)


if __name__ == '__main__':
    # example usage
    from data_filterer import DataFilterer

    filterer_instance = DataFilterer(model=Resnet32Example(),
                                     layer="avgpool",  # see print(model) for layer names
                                     device="cuda")

    idxs = filterer_instance.get_idxs(semantic_percentage=0.09,
                                      outlier_percentage=0.01,
                                      downscale_dim=None,  # no dimensionality reduction to maximize accuracy
                                      downscale_method="PCA",
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
