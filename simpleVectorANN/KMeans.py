import time

from joblib import Parallel, delayed

import numpy as np

class KMeans:
    def __init__(self, n_clusters=10, n_dimensions=128, max_iters=10):
        self.n_dimensions = n_dimensions
        self.n_clusters = n_clusters
        self.data = np.empty((0, n_dimensions))
        self.fitted = False
        self.max_iters = max_iters

        self.centroids = None
        self.cluster_members = None

    def _initialize_centroids(self):
        indices = np.random.choice(len(self.data), self.n_clusters, replace=False)
        self.centroids = self.data[indices]

    def _find_closest_centroid_batch(self, j, N, job_size):
        ret = []
        for i in range(j * job_size, min(N, (j + 1) * job_size)):
            point = self.data[i]
            distances = np.linalg.norm(point - self.centroids, axis=1)
            nearest_centroid = np.argmin(distances)
            ret.append((nearest_centroid, i))
        return ret

    def _assign_clusters_parallel(self, n_jobs=32):
        self.cluster_members = [[] for _ in range(self.n_clusters)]
        N = len(self.data)
        job_size = int(np.ceil(N / n_jobs))
        return_assignments = Parallel(n_jobs=n_jobs)(delayed(self._find_closest_centroid_batch)(j, N, job_size) for j in range(n_jobs))
        for nearest_centroid, point_index in np.concat(return_assignments):
            self.cluster_members[nearest_centroid].append(point_index)

    def _assign_clusters(self):
        self.cluster_members = [[] for _ in range(self.n_clusters)]
        for i, point in enumerate(self.data):
            distances = np.linalg.norm(point - self.centroids, axis=1)
            nearest_centroid = np.argmin(distances)
            self.cluster_members[nearest_centroid].append(i)

    def _update_centroids(self):
        for i in range(self.n_clusters):
            if len(self.cluster_members[i]) > 0:
                points_in_cluster_i = self.data[self.cluster_members[i]]
                self.centroids[i] = np.mean(points_in_cluster_i, axis=0)

    def fit(self, tolerance=1e-6):
        if self.n_clusters > len(self.data):
            raise ValueError(
                f"Too many centroids: Cannot cluster {len(self.data)} points with {self.n_clusters} centers.")
        if self.fitted:
            pass

        self._initialize_centroids()
        converged = False
        for i in range(self.max_iters):
            old_centroids = np.copy(self.centroids)
            print(f"Running iteration {i}")
            start_time = time.time()
            self._assign_clusters_parallel()
            elapsed_time = time.time() - start_time
            print(f"Assigned clusters in: {elapsed_time}")
            start_time = time.time()
            self._update_centroids()
            elapsed_time = time.time() - start_time
            print(f"Updated Centroids in: {elapsed_time}")
            centroid_shift = np.mean(np.linalg.norm(self.centroids - old_centroids, axis=1))
            print(f"Overall (average) centroid shift: {centroid_shift}")
            if centroid_shift < tolerance:
                print("Converged")
                converged = True
                break
        if not converged:
            print(f"Failed to converge after {self.max_iters} iterations.")

        self.fitted = True
    
    def add_points(self, X):
        if X.shape[1] != self.n_dimensions:
            raise ValueError(f"Trying to insert data with {X.shape[1]} dimensions, must be {self.n_dimensions}")
        self.data = np.concat((self.data, X), axis=0)
    
    def queryANN(self, query_point, num_NN=10, num_centroids_to_check=1):
        distances_to_centroids = np.linalg.norm(query_point - self.centroids, axis=1)
        nearest_centroid_indices = np.argsort(distances_to_centroids)

        data_subset_to_search = np.concat([
            self.cluster_members[centroid_index]
            for centroid_index in nearest_centroid_indices[:num_centroids_to_check]
        ])
        distances_to_subset_elements = np.linalg.norm(query_point - self.data[data_subset_to_search], axis=1)
        distance_index_tuples = zip(distances_to_subset_elements, data_subset_to_search)
        return [i for _, i in sorted(distance_index_tuples)[:num_NN]]
