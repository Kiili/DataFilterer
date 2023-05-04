import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from filterable_interface import FilterableInterface


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self):
        super().__init__(root='./data',
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])]))

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        img, label = super().__getitem__(index=idx)

        # Return the image and the index of the image as metadata
        return img, idx


class Resnet32Example(FilterableInterface):

    def __init__(self):
        # A pretrained model on CIFAR10
        self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)

        self.filterable_dataset = DataLoader(dataset=CIFAR10Dataset(), batch_size=128)

        # dataset without transformations for viewing images
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

    def get_model(self) -> nn.Module:
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
                                     device="cpu")

    idxs = filterer_instance.get_idxs(outliar_percentage=0.1,
                                      semantic_percentage=0)
    # print(idxs)  # the indexes of images that are in the filtered dataset


    # filterer_instance.get_imgs((640, 640))

    import matplotlib as mpl
    mpl.use("GTK3Agg")
    plt = filterer_instance.get_plot()
    plt.show()
