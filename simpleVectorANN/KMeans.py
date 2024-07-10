
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

    def fit(self):
        if self.n_clusters > len(self.data):
            raise ValueError(
                f"Too many centroids: Cannot cluster {len(self.data)} points with {self.n_clusters} centers.")
        
        if self.fitted:
            pass

        self._initialize_centroids()
        for i in range(self.max_iters):
            old_centroids = np.copy(self.centroids)
            print(f"Running iteration {i}")
            self._assign_clusters()
            self._update_centroids()
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            print(f"Overall centroid shift: {centroid_shift}")
    
    def add_points(self, X):
        if X.shape[1] != self.n_dimensions:
            raise ValueError(f"Trying to insert data with {X.shape[1]} dimensions, must be {self.n_dimensions}")
        self.data = np.concat((self.data, X), axis=0)