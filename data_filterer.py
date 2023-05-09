import math
import time
from typing import Union

import matplotlib
import torch

from filterable_interface import FilterableInterface
from database import Cluster_filterer


class DataFilterer:
    def __init__(self,
                 model: FilterableInterface,
                 layer: str,
                 device: str | torch.device | None = None):
        """
        :param model: Class instance that extends FilterableInterface
        See FilterableInterface contents for details

        :param layer: layer name of the model that is used to extract feature maps from
        See print(model.get_model()) for model naming tree

        :param device: device to use for filtering and dataset forwarding
        cpu is experimental and usable in small datasets and models
        By default, cuda is used if available
        """
        self.model = model
        self.layer = layer
        self.data_args = []
        self.passed_idxs, self.failed_idxs, self.plot_points, self.db = None, None, None, None
        if not device:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device(device)

        self.model.get_model().to(self.device)
        self.model.get_model().eval()

        self.activation = []
        self.hook = self.model.get_model().get_submodule(self.layer).register_forward_hook(
            lambda model, input, output: self.activation.append(
                output if torch.is_tensor(output) else torch.stack(output)))

    def eval_at_layer(self, data):
        """
        Evaluate the model at a layer
        :param data: batch of data
        :return: batch of extracted feature maps
        """
        self.activation = []
        with torch.no_grad():
            self.model.get_model()(data)
        if len(self.activation) == 0:
            return torch.Tensor()
        return self.activation[0].detach()

    def get_dataset_info(self):
        """
        Does a full pass on the dataset
        :return: amount of items and the amount of batches in the dataset
        """
        items = 0
        batches = 0
        for batch in self.model.get_filterable_dataset():
            batch, args = batch
            items += len(batch)
            batches += 1
        return items, batches

    def get_feature_map_shape(self):
        """
        :return: shape of the feature map that will be extracted from the model
        """
        for batch in self.model.get_filterable_dataset():
            batch, args = batch

            if torch.is_tensor(batch):
                batch = batch.to(self.device)

            layer_data = self.eval_at_layer(batch).cpu().numpy()
            return layer_data.shape[1:]

    def calc_db(self):
        """
        Extract the feature maps from the filterable dataset
        """
        if self.db is not None:
            self.db.__del__()
        self.db = Cluster_filterer(device=self.device)
        self.data_args = []
        start = time.time()

        for i, batch in enumerate(self.model.get_filterable_dataset()):
            batch, args = batch

            self.data_args += [arg for arg in args]

            if torch.is_tensor(batch):
                batch = batch.to(self.device)

            layer_data = self.eval_at_layer(batch)
            layer_data = layer_data.reshape(len(batch), -1)

            self.db.insert(layer_data)

            print(f"\rbatch {i + 1}, time: {round(time.time() - start, 2)}s, "
                  f"{round((time.time() - start) / (i + 1), 2)}s/batch",
                  flush=True, end="")
        print("")

    def get_idxs(self,
                 semantic_percentage: float,
                 outlier_percentage: float,
                 downscale_dim: Union[int, None] = 50,
                 downscale_method: str = "PCA") -> list:
        """
        :param semantic_percentage:
        Fraction (0.0-1.0) to remove semantically similar images

        :param outlier_percentage:
        Fraction (0.0-1.0) to remove outliers

        :param downscale_dim:
        Parameter that specifies the dimensionality of the data during filtering
        Higher values increase the accuracy of the filtering but take more time and memory
        0 or None indicate maximal accuracy (downscale_dim=math.inf)
        It is overridden to min(downscale_dim, n_dataset_items, n_feature_map_size)

        :param downscale_method:
        The method to reduce the dimensionality of the feature maps.
        Supported are:
        "PCA": Principal Component Analysis
        "UMAP": Uniform Manifold Approximation and Projection
        "T-SVD": Truncated SVD
        "SRP": Sparse Random Projection
        "GRP": Gaussian Random Projection

        :return: list of args, which describe the data that is kept in the dataset
        """
        if semantic_percentage + outlier_percentage >= 1:
            return []
        self.calc_db()
        self.passed_idxs, self.failed_idxs, self.plot_points = \
            self.db.get_idxs(outlier_percentage=outlier_percentage,
                             semantic_percentage=semantic_percentage,
                             downscale_dim=math.inf if (downscale_dim in (0, None)) else downscale_dim,
                             downscale_method=downscale_method)
        self.db.__del__()
        return [self.data_args[idx] for idx in self.passed_idxs]

    def get_filtered_out(self) -> tuple[dict, list]:
        """
        :similars:
        A dictionary where keys represent items that were included in the filtered dataset
        Values represent lists of items in the dataset that were classified as too semantically similar

        :outliers:
        List of detected outliers in the dataset

        :return: similars, outliers
        """
        if not self.failed_idxs:
            return {}, []
        outliers = []
        similars = dict()
        for k, v in self.failed_idxs.items():
            if v == [-1]:
                outliers.append(self.data_args[k])
                continue
            similars[self.data_args[k]] = [self.data_args[x] for x in v]
        return similars, outliers

    def get_fig(self, plt):
        """
        :param plt:
        from matplotlib import pyplot as plt
        get_fig(plt)
        plt.show()

        :return: figure that describes the dataset in 3D space
        """
        print("plotting", flush=True)
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        colors = []

        passed = set(self.passed_idxs)

        for i in range(len(self.plot_points)):
            if i in passed:
                colors.append((0, 1, 0, 0.3))
                continue

            if i in self.failed_idxs and self.failed_idxs[i] == [-1]:
                colors.append((1, 0, 0, 1))
                continue

            colors.append((0, 0, 1, 1))

        ax.scatter3D([a[0] for a in self.plot_points],
                     [a[1] for a in self.plot_points],
                     [a[2] for a in self.plot_points],
                     c=colors)

        for k, v in self.failed_idxs.items():
            if v != [-1]:
                for item in v:
                    ax.plot([self.plot_points[k][0], self.plot_points[item][0]],
                            [self.plot_points[k][1], self.plot_points[item][1]],
                            zs=[self.plot_points[k][2], self.plot_points[item][2]])
        fig.tight_layout()
        return fig

    def __del__(self):
        self.hook.remove()
        self.data_args = []
        if self.db:
            self.db.__del__()
        self.db = None
        self.activation = []
