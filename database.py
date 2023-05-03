import time

import numpy as np
import sklearn
import torch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import hdbscan
import tensordict.memmap as mem


class Cluster_filterer:

    def __init__(self, dim_size, device, max_items=None):
        self.dim_size = dim_size
        self.device = torch.device(device)

        # cache feature maps to disk instead of (v)ram
        # self.cache = "cache.pt"
        # self.db = np.memmap(shape=(max_items, dim_size), filename=self.cache, mode="w+", dtype=float)
        # self.db = mem.MemmapTensor(2500, dim_size, filename=self.cache, mode="r+", device=self.device)
        # self.item_idx = 0
        self.db = torch.Tensor().to(self.device)

    def insert(self, x: torch.tensor):
        # todo store not in ram or vram, but disk
        x = x.to(self.device)  # .cpu().numpy()
        # for b in x:
        #     self.db[self.item_idx] = b
        #     self.item_idx += 1
        self.db = torch.cat((self.db, x), dim=0)

    def pca_incremental_cuda(self, features, batch_size, n_components=3):
        print("pca incremental ", end="", flush=True)
        s = time.time()
        import cuml
        import cudf

        pca = cuml.IncrementalPCA(n_components=n_components,
                                  batch_size=batch_size,
                                  output_type="numpy",
                                  copy=True)

        for batch in cuml.decomposition.incremental_pca._gen_batches(features.shape[0], batch_size,
                                                                     min_batch_size=n_components or 0):
            X_batch = features[batch]
            pca.partial_fit(X_batch)

        output = []
        for batch in cuml.decomposition.incremental_pca._gen_batches(features.shape[0], batch_size,
                                                                     min_batch_size=n_components or 0):
            output.append(pca.transform(features[batch]))
        import cupy as cp
        output, _, _, _ = cuml.internals.input_utils.input_to_cuml_array(cp.vstack(output), order='K')

        print(round(time.time() - s, 2), flush=True)
        return cudf.DataFrame(output)

    def pca_incremental_cpu(self, features, batch_size, n_components=3):
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        out = pca.fit_transform(features)
        return out

    def pca(self, n_components=3, niter=50):
        print("pca ", end="", flush=True)
        s = time.time()
        if not torch.is_tensor(self.db):
            self.db = torch.tensor(self.db, device=self.device)
        U, _, _ = torch.pca_lowrank(A=self.db.to(self.device), q=n_components, niter=niter)
        print(round(time.time() - s, 2), flush=True)
        return U

    def plot_points(self, pca_arr, idxs, not_passed):
        print("plotting", flush=True)
        mpl.use("GTK3Agg")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = []

        for i in range(len(pca_arr)):
            if i in idxs:
                colors.append((0, 1, 0, 0.3))
                continue

            if i in not_passed and not_passed[i] == [-1]:
                colors.append((1, 0, 0, 1))
                continue

            colors.append((0, 0, 1, 1))

        ax.scatter3D([a[0] for a in pca_arr], [a[1] for a in pca_arr], [a[2] for a in pca_arr],
                     c=colors)
        for k, v in not_passed.items():
            if v != [-1]:
                for item in v:
                    ax.plot([pca_arr[k][0], pca_arr[item][0]],
                            [pca_arr[k][1], pca_arr[item][1]],
                            zs=[pca_arr[k][2], pca_arr[item][2]])

        plt.savefig("plt.jpg")
        plt.show()

    def hdbscan_cuda(self, cluster_size, not_passed):
        import cuml
        import cudf
        print("hdbscan_cuda ", flush=True, end="")
        if type(self.db) is not cudf.DataFrame:
            self.db = cudf.DataFrame(self.db)
        hdb = cuml.cluster.HDBSCAN(min_cluster_size=cluster_size,
                                   allow_single_cluster=True,
                                   output_type="numpy",
                                   min_samples=2,
                                   cluster_selection_method="eom",
                                   )

        s = time.time()
        hdb.fit(self.db)
        print(round(time.time() - s, 2), flush=True)
        is_in = [i for i in range(len(hdb.labels_)) if hdb.labels_[i] == 0]
        for i, b in enumerate(hdb.labels_):
            if b != 0:
                not_passed[i] = [-1]

        self.db = self.db.take(is_in)

        return is_in

    def hdbscan_cpu(self, cluster_size, not_passed):
        print("hdbscan_cpu ", flush=True, end="")
        if torch.is_tensor(self.db):
            self.db = self.db.cpu().numpy()
        if type(self.db) is not np.ndarray:
            self.db = np.array(self.db)
        hdb = hdbscan.HDBSCAN(min_cluster_size=cluster_size,
                              allow_single_cluster=True,
                              min_samples=2,
                              )
        s = time.time()
        hdb.fit(self.db)
        print(round(time.time() - s, 2), flush=True)
        for i, b in enumerate(hdb.labels_):
            if b != 0:
                not_passed[i] = [-1]
        is_in = [i for i in range(len(hdb.labels_)) if hdb.labels_[i] == 0]
        self.db = self.db[is_in]
        return is_in

    def agg_cluster_cuda(self, num_clusters):
        import cuml
        import cudf
        print("clustering_cuda ", flush=True, end="")
        if type(self.db) is not cudf.DataFrame:
            self.db = cudf.DataFrame(self.db)
        clusterer = cuml.AgglomerativeClustering(n_clusters=num_clusters,
                                                 affinity="cosine",
                                                 linkage="single",
                                                 output_type="numpy",
                                                 connectivity="knn",
                                                 n_neighbors=2,
                                                 )

        s = time.time()
        clusterer.fit(self.db)
        print(round(time.time() - s, 2), flush=True)
        return clusterer.labels_

    def agg_cluster_cpu(self, num_clusters):
        print("clustering_cpu ", flush=True, end="")
        if torch.is_tensor(self.db):
            self.db = self.db.cpu().numpy()
        if type(self.db) is not np.ndarray:
            self.db = self.db.to_numpy()
        clusterer = AgglomerativeClustering(n_clusters=num_clusters,
                                            metric="cosine",
                                            linkage="single",
                                            )
        s = time.time()
        clusterer.fit(self.db)
        print(round(time.time() - s, 2), flush=True)
        return clusterer.labels_

    def get_cluster_center_idxs(self, cluster_labels, num_clusters, db_idxs, not_passed):
        if type(self.db) is not np.ndarray:
            self.db = self.db.to_numpy()

        nc = NearestCentroid(metric="euclidean")
        nc.fit(self.db, cluster_labels)
        cluster_centers = nc.centroids_

        dists = np.full(shape=(num_clusters,), fill_value=np.inf, dtype=float)
        idxs = np.zeros(shape=(num_clusters,), dtype=int)

        for i, cluster_idx in enumerate(cluster_labels):
            dist = cosine_distances(self.db[i].reshape(1, -1), cluster_centers[cluster_idx].reshape(1, -1))[0]
            if dist < dists[cluster_idx]:
                if dists[cluster_idx] != np.inf:
                    not_passed[db_idxs[i]] = not_passed.get(db_idxs[i], []) + [idxs[cluster_idx]] + not_passed.get(
                        idxs[cluster_idx], [])
                    if idxs[cluster_idx] in not_passed:
                        del not_passed[idxs[cluster_idx]]
                dists[cluster_idx] = dist
                idxs[cluster_idx] = db_idxs[i]
            else:
                not_passed[idxs[cluster_idx]] = not_passed.get(idxs[cluster_idx], []) + [db_idxs[i]]

        return set(idxs)

    def get_idxs(self, plot_points, outliar_percentage, semantic_percentage):
        # self.db = self.db[:self.item_idx + 1]
        downscale_dim = min(*self.db.shape, 100)
        print(self.db.shape)

        if plot_points:
            pca_arr = self.pca(3)
            plot_points = pca_arr.cpu().numpy()
            # self.db = pca_arr
            # plot_points = self.pca_incremental_cpu(features=self.db, n_components=3, batch_size=5)
            # plot_points = self.pca_incremental_cuda(features=self.db,
            #                                        n_components=3,
            #                                        batch_size=10).to_numpy()

        if downscale_dim < self.db.shape[1]:
            # downscale the feature maps
            # self.db = self.pca_incremental_cuda(features=self.db,
            #                                     n_components=downscale_dim,
            #                                     batch_size=10)
            self.db = self.pca(n_components=downscale_dim)

        not_passed = dict()
        out_dataset_size = int((1 - outliar_percentage - semantic_percentage) * self.db.shape[0])
        db_idxs = np.array(range(self.db.shape[0]), dtype=int)

        # outliar detection
        cluster_size = int(self.db.shape[0] * (1 - outliar_percentage))
        is_in = self.hdbscan_cuda(cluster_size, not_passed)
        db_idxs = db_idxs[is_in]

        # clustering
        num_clusters = out_dataset_size
        cluster_labels = self.agg_cluster_cuda(num_clusters)
        idxs = self.get_cluster_center_idxs(cluster_labels, num_clusters, db_idxs, not_passed)

        # plotting
        if plot_points is not False:
            self.plot_points(plot_points, idxs, not_passed)

        return idxs, not_passed

    def __del__(self):
        self.db = torch.Tensor().to(self.device)
