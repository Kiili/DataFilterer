from typing import Collection

import numpy as np
from torch import nn


class FilterableInterface:

    def get_model(self) -> nn.Module:
        """
        :return: Pytorch model
        """
        raise NotImplementedError

    def get_filterable_dataset(self) -> Collection:
        """
        batch: 1 batch of input to model. Batch contents must be with constant shape

        args: any metadata for each element in :batch:

        :return: Iterable with __len__ (Collection). Each item with shape (batch, args)
        """
        raise NotImplementedError

    def get_image(self, arg) -> np.ndarray:
        """
        optional for show_imgs
        :return: a single image from the filterable dataset specified by its :arg:
        """
        raise NotImplementedError
