import math
import time

import cv2
import numpy as np
import torch

import filterable_interface
from database import Cluster_filterer


class DataFilterer:

    def __init__(self, model: filterable_interface.FilterableInterface, layer, device=None):
        self.model = model
        self.layer = layer
        self.db = None
        self.data_args = []
        self.passed_idxs, self.failed_idxs, self.plot_points = None, None, None
        if not device:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model.get_model().to(self.device)
        self.model.get_model().eval()

        self.activation = []
        self.hook = self.model.get_model().get_submodule(self.layer).register_forward_hook(
            lambda model, input, output: self.activation.append(
                output if torch.is_tensor(output) else torch.stack(output)))

    def eval_at_layer(self, data):
        self.activation = []
        with torch.no_grad():
            self.model.get_model()(data)
        if len(self.activation) == 0:
            return torch.Tensor()
        return self.activation[0].detach()

    def get_dataset_info(self):
        items = 0
        batches = 0
        for batch in self.model.get_filterable_dataset():
            batch, args = batch
            items += len(batch)
            batches += 1
        return items, batches

    def get_feature_map_shape(self):
        for batch in self.model.get_filterable_dataset():
            batch, args = batch
            batch = batch.to(self.device)
            layer_data = self.eval_at_layer(batch).cpu().numpy()
            return layer_data.shape[1:]

    def calc_db(self):
        self.db = None
        self.data_args = []
        s = time.time()
        start = time.time()

        for i, batch in enumerate(self.model.get_filterable_dataset()):
            batch, args = batch
            batch = batch.to(self.device)

            self.data_args += [arg for arg in args]

            batch_size = len(batch)
            layer_data = self.eval_at_layer(batch)
            layer_data = layer_data.reshape(batch_size, -1)

            if not self.db:
                print("dim size:", layer_data.shape)
                self.db = Cluster_filterer(dim_size=layer_data.shape[-1], device=self.device,
                                           max_items=len(self.model.get_filterable_dataset()) * batch_size)

            self.db.insert(layer_data)

            print(
                f"\rbatch {i + 1}, time: {round(time.time() - s, 3)}, {round((time.time() - start) / (i + 1), 2)}s/batch",
                flush=True, end="")
            s = time.time()
        print("")

    def get_idxs(self,
                 outliar_percentage: float,
                 semantic_percentage: float):
        if not self.db:
            self.calc_db()
        self.passed_idxs, self.failed_idxs, self.plot_points = self.db.get_idxs(outliar_percentage=outliar_percentage,
                                                                                semantic_percentage=semantic_percentage)

        return [self.data_args[idx] for idx in self.passed_idxs]

    def get_imgs(self, resolution: tuple):

        cv2.destroyAllWindows()
        for k, v in sorted(self.failed_idxs.items(), key=lambda x: -len(x[1])):
            img = self.model.get_image(self.data_args[k])
            img = cv2.resize(np.array(img), resolution)
            # print(self.data_args[k], end=", ")
            if v != [-1]:
                similars = []
                for i in v:
                    similars.append(self.model.get_image(self.data_args[i]))
                    # print(self.data_args[i], end=", ")
                square_size = math.ceil(math.sqrt(len(similars)))
                rows = []
                for i in range(square_size):
                    row = []
                    for j in range(square_size):
                        if len(similars) > i * square_size + j:
                            row.append(similars[i * square_size + j])
                        else:
                            row.append(np.zeros_like(similars[0]))
                    rows.append(np.concatenate(row, axis=1))
                img2 = np.concatenate(rows, axis=0)
                img2 = cv2.resize(img2, resolution)
                img = np.concatenate((img, img2), axis=1)

            cv2.imshow("out", img)
            cv2.resizeWindow("out", resolution[0] * 2, resolution[1])
            cv2.moveWindow("out", 1920, 100)
            # print("")
            cv2.waitKey(0)

    def get_plot(self):
        from matplotlib import pyplot as plt
        print("plotting", flush=True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = []

        for i in range(len(self.plot_points)):
            if i in self.passed_idxs:
                colors.append((0, 1, 0, 0.3))
                continue

            if i in self.failed_idxs and self.failed_idxs[i] == [-1]:
                colors.append((1, 0, 0, 1))
                continue

            colors.append((0, 0, 1, 1))

        ax.scatter3D([a[0] for a in self.plot_points], [a[1] for a in self.plot_points], [a[2] for a in self.plot_points],
                     c=colors)
        for k, v in self.failed_idxs.items():
            if v != [-1]:
                for item in v:
                    ax.plot([self.plot_points[k][0], self.plot_points[item][0]],
                            [self.plot_points[k][1], self.plot_points[item][1]],
                            zs=[self.plot_points[k][2], self.plot_points[item][2]])

        return plt

    def __del__(self):
        self.hook.remove()
        self.data_args = []
        if self.db:
            self.db.__del__()
        self.db = None
        self.activation = []
