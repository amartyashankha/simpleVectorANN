import os 

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .load_data import download_dataset
from .KMeans import KMeans

def query_NN(query_point, kmeans):
    return kmeans.queryANN(query_point)

def main():
    hdf5_filename = "sift.hdf5"
    if not os.path.exists(hdf5_filename):
        download_dataset(hdf5_filename)
    
    dataset = h5py.File(hdf5_filename, "r")

    points = np.array(dataset["points"])
    queries = np.array(dataset["queries"])
    nearest_neighbors = np.array(dataset["nearest_neighbors"])
    dimension = dataset.attrs["dimension"]

    kmeans = KMeans(n_clusters=100, n_dimensions=128, max_iters=100)
    kmeans.add_points(points)
    kmeans.fit(tolerance=10)

    # Truncating queries to evaluate faster
    queries = queries[:100]
    evaluated_NN_values = Parallel(n_jobs=32)(delayed(query_NN)(query_point, kmeans) for query_point in tqdm(queries))
    print("Finished evaluating queries")
    print("Computing recall@10 . . .")

    recall_at_10_accumulator = 0
    for evaluated_NN, groundtruth_NN in zip(evaluated_NN_values, nearest_neighbors):
        groundtruth_NN_truncated = groundtruth_NN[:10]
        recall_at_10 = len(set(evaluated_NN) & set(groundtruth_NN_truncated)) / 10
        recall_at_10_accumulator += recall_at_10
    
    print(f"Average recall@10: {recall_at_10_accumulator / len(queries)}")


if __name__ == "__main__":
    main()