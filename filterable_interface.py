from typing import Iterable
import numpy as np
from torch import nn


class FilterableInterface:

    def get_model(self) -> nn.Module:
        """
        :return: pytorch model
        """
        raise NotImplementedError

    def get_filterable_dataset(self) -> Iterable:
        """
        batch: 1 batch of input to model. Batch contents must be with constant shape

        args: any metadata for each element in :batch:

        :return: Iterable. Each item like (batch, args)
        """
        raise NotImplementedError

    def get_image(self, arg) -> np.ndarray:
        """
        optional for show_imgs
        :return: single image from the filterable dataset specified by its :arg:
        """
        raise NotImplementedError
