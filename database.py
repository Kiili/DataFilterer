import time
import warnings

import numpy as np
import torch
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import IncrementalPCA


class Cluster_filterer:

    def __init__(self, device):
        self.device = torch.device(device)

        # cache feature maps to disk instead of (v)ram
        # self.db = np.memmap(shape=(max_items, dim_size), filename="cache.pt", mode="w+", dtype=float)
        # self.item_idx = 0

        # cache feature maps to (v)ram
        self.db = torch.Tensor().to(self.device)

    def insert(self, x: torch.tensor):
        # cache feature maps to (v)ram
        x = x.to(self.device)
        self.db = torch.cat((self.db, x), dim=0)

        # cache feature maps to disk
        # x = x.cpu().numpy()
        # for batch in x:
        #     self.db[self.item_idx] = batch
        #     self.item_idx += 1

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

    def pca(self, n_components, niter=50):
        print("pca ", end="", flush=True)
        s = time.time()
        if not torch.is_tensor(self.db):
            self.db = torch.tensor(self.db, device=self.device)
        U, _, _ = torch.pca_lowrank(A=self.db.to(self.device), q=n_components, niter=niter)
        print(round(time.time() - s, 2), flush=True)
        return U

    def hdbscan_cuda(self, cluster_size, not_passed):
        import cuml
        print("hdbscan_cuda ", flush=True, end="")
        # maybe the only doc about this
        # https://gitee.com/mirrors/cuML/blob/branch-23.06/python/cuml/cluster/hdbscan/hdbscan.pyx
        hdb = cuml.cluster.HDBSCAN(min_cluster_size=cluster_size,
                                   max_cluster_size=cluster_size,
                                   allow_single_cluster=True,
                                   min_samples=2,
                                   cluster_selection_method="eom",
                                   output_type="numpy",
                                   connectivity='knn',
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
        hdb = hdbscan.HDBSCAN(min_cluster_size=cluster_size,
                              max_cluster_size=cluster_size,
                              allow_single_cluster=True,
                              cluster_selection_method="eom",
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
        print("clustering_cuda ", flush=True, end="")
        clusterer = cuml.AgglomerativeClustering(n_clusters=num_clusters,
                                                 affinity="cosine",
                                                 linkage="single",
                                                 output_type="numpy",
                                                 connectivity="knn",
                                                 )
        s = time.time()
        clusterer.fit(self.db)
        print(round(time.time() - s, 2), flush=True)
        return clusterer.labels_

    def agg_cluster_cpu(self, num_clusters):
        """
        This is O(n**2) memory, because connectivity="knn" is not supported
        """
        print("clustering_cpu ", flush=True, end="")
        clusterer = AgglomerativeClustering(n_clusters=num_clusters,
                                            metric="cosine",
                                            affinity="cosine",
                                            linkage="single",
                                            )
        s = time.time()
        clusterer.fit(self.db)
        print(round(time.time() - s, 2), flush=True)
        return clusterer.labels_

    def get_cluster_center_idxs(self, cluster_labels, num_clusters, db_idxs, not_passed):
        """
        Find the center-most feature map of each cluster
        :param cluster_labels: cluster label for each feature map
        :param num_clusters: total amount of clusters
        :param db_idxs: the original dataset indexes for each feature map

        :param not_passed: fill this dict(k, v) with
        k: center-most feature map index in a cluster
        v: [other feature map indexes that are in the same cluster as k]

        :return: indexes of feature maps that are kept in the dataset
        """
        print("cluster_centroids ", flush=True, end="")
        s = time.time()
        nc = NearestCentroid(metric="cosine")
        with warnings.catch_warnings(record=True):
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
        print(round(time.time() - s, 2), flush=True)
        return set(idxs)

    def get_idxs(self, outliar_percentage, semantic_percentage):
        f = self.get_idxs_cuda
        if self.device == torch.device("cpu"):
            f = self.get_idxs_cpu

        return f(outliar_percentage, semantic_percentage)

    def get_idxs_cpu(self, outliar_percentage, semantic_percentage):
        downscale_dim = min(*self.db.shape, 100)
        plot_points = self.pca(3).cpu().numpy()

        if downscale_dim < self.db.shape[1]:
            self.db = self.pca(n_components=downscale_dim)

        not_passed = dict()
        out_dataset_size = int((1 - outliar_percentage - semantic_percentage) * self.db.shape[0])
        db_idxs = np.array(range(self.db.shape[0]), dtype=int)

        # outlier detection
        cluster_size = int(self.db.shape[0] * (1 - outliar_percentage))
        self.db = self.db.numpy()
        is_in = self.hdbscan_cpu(cluster_size, not_passed)
        db_idxs = db_idxs[is_in]

        # clustering
        num_clusters = out_dataset_size
        cluster_labels = self.agg_cluster_cpu(num_clusters)
        idxs = self.get_cluster_center_idxs(cluster_labels, num_clusters, db_idxs, not_passed)

        return idxs, not_passed, plot_points

    def get_idxs_cuda(self, outliar_percentage, semantic_percentage):
        import cudf
        downscale_dim = min(*self.db.shape, 100)
        plot_points = self.pca(3).cpu().numpy()

        if downscale_dim < self.db.shape[1]:
            self.db = self.pca(n_components=downscale_dim)

        not_passed = dict()
        out_dataset_size = int((1 - outliar_percentage - semantic_percentage) * self.db.shape[0])
        db_idxs = np.array(range(self.db.shape[0]), dtype=int)

        # outlier detection
        cluster_size = int(self.db.shape[0] * (1 - outliar_percentage))
        self.db = cudf.DataFrame(self.db)
        is_in = self.hdbscan_cuda(cluster_size, not_passed)
        db_idxs = db_idxs[is_in]

        # clustering
        num_clusters = out_dataset_size
        cluster_labels = self.agg_cluster_cuda(num_clusters)
        self.db = self.db.to_numpy()
        idxs = self.get_cluster_center_idxs(cluster_labels, num_clusters, db_idxs, not_passed)

        return idxs, not_passed, plot_points

    def __del__(self):
        self.db = torch.Tensor().to(self.device)
