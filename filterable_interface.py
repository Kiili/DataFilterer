from typing import Iterable
import numpy as np
import torch


class FilterableInterface:

    def get_model(self) -> torch.nn.Module:
        """
        :return: pytorch model
        """
        raise NotImplementedError

    def get_filterable_dataset(self) -> Iterable:
        """
        batch: 1 batch of input to model.
        Batch contents must be with constant shape

        args: any metadata for each element in :batch:

        :return: Iterable. Each item like (batch, args)
        """
        raise NotImplementedError

    def get_image(self, arg) -> np.ndarray:
        """
        optional for viewing images in the application

        If :arg: is used as image file path, it can be implemented:
        PIL:    np.array(PIL.Image.open(arg))
        OpenCV: cv2.imread(arg)[:,:,::-1]  # BGR to RGB

        :return: channels-last RGB image from the
        filterable dataset specified by its :arg:
        """
        raise NotImplementedError
