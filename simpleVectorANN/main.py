import os 
import argparse

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .load_data import download_dataset
from .KMeans import KMeans

def main():
    parser = argparse.ArgumentParser(description='ANN based on inverted indices')
    parser.add_argument('-K', '--num-clusters', type=int, help='Number of K-means clusters', required=False, default=100)
    parser.add_argument('-P', '--num-clusters-to-search', type=int, help='Number of nearest clusters to search', required=False, default=3)
    parser.add_argument('-t', '--tolerance', type=float, help='Tolerance for K-means convergence', required=False, default=10)

    args = parser.parse_args()


    hdf5_filename = "sift.hdf5"
    if not os.path.exists(hdf5_filename):
        download_dataset(hdf5_filename)
    
    dataset = h5py.File(hdf5_filename, "r")

    points = np.array(dataset["points"])
    queries = np.array(dataset["queries"])
    nearest_neighbors = np.array(dataset["nearest_neighbors"])
    dimension = dataset.attrs["dimension"]

    kmeans = KMeans(n_clusters=args.num_clusters, n_dimensions=dimension, max_iters=100)
    kmeans.add_points(points)
    kmeans.fit(tolerance=args.tolerance)

    # Truncating queries to evaluate faster
    queries = queries[:1000]
    print("Evaluating queries")
    evaluated_NN_values = [
        kmeans.queryANN(query_point, num_centroids_to_check=args.num_clusters_to_search)
        for query_point in tqdm(queries)
        ]
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