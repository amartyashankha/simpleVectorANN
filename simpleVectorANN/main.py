import os 

import h5py
import numpy as np

from .load_data import download_dataset
from .KMeans import KMeans

def main():
    hdf5_filename = "sift.hdf5"
    if not os.path.exists(hdf5_filename):
        download_dataset(hdf5_filename)
    
    dataset = h5py.File(hdf5_filename, "r")

    points = np.array(dataset["points"])
    queries = np.array(dataset["queries"])
    nearest_neighbors = np.array(dataset["nearest_neighbors"])
    dimension = dataset.attrs["dimension"]

    kmeans = KMeans(n_clusters=10, n_dimensions=128, max_iters=100)
    kmeans.add_points(points[:1000000])
    kmeans.fit(tolerance=10)

    recall_at_10_accumulator = 0
    for query_point, groundtruth_NN in zip(queries, nearest_neighbors):
        evaluated_NN = kmeans.queryANN(query_point)
        recall_at_10 = len(set(evaluated_NN) & set(groundtruth_NN)) / 10
        if recall_at_10 != 1.0:
            print(recall_at_10)
        recall_at_10_accumulator += recall_at_10
    
    print(f"Average recall@10: {recall_at_10_accumulator / len(queries)}")


if __name__ == "__main__":
    main()