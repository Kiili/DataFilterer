import time
import torch

from filterable_interface import FilterableInterface
from database import Cluster_filterer


class DataFilterer:
    def __init__(self,
                 model: FilterableInterface,
                 layer: str,
                 device: str | torch.device | None = None):
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

            layer_data = self.eval_at_layer(batch.to(self.device))
            layer_data = layer_data.reshape(len(batch), -1)

            self.db.insert(layer_data)

            print(f"\rbatch {i + 1}, time: {round(time.time() - start, 2)}s, "
                  f"{round((time.time() - start) / (i + 1), 2)}s/batch",
                  flush=True, end="")
        print("")

    def get_idxs(self,
                 semantic_percentage: float,
                 outliar_percentage: float):
        """
        :param semantic_percentage: Fraction to remove semantically similar images
        :param outliar_percentage: Fraction to remove outliers
        :return: list of args, which describe the data that is kept in the dataset
        """
        self.calc_db()
        self.passed_idxs, self.failed_idxs, self.plot_points = self.db.get_idxs(outliar_percentage=outliar_percentage,
                                                                                semantic_percentage=semantic_percentage)

        return [self.data_args[idx] for idx in self.passed_idxs]

    def get_fig(self, plt):
        """
        :param plt: matplotlib.pyplot instance
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
