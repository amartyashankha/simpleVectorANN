import os 

import h5py
import numpy as np

from .load_data import download_dataset

def main():
    hdf5_filename = "sift.hdf5"
    if not os.path.exists(hdf5_filename):
        download_dataset(hdf5_filename)
    
    dataset = h5py.File(hdf5_filename, "r")

    points = np.array(dataset["points"])
    queries = np.array(dataset["queries"])
    nearest_neighbors = np.array(dataset["nearest_neighbors"])
    dimension = dataset.attrs["dimension"]

    query_point = queries[0]
    distance_vectors = points - query_point
    print(distance_vectors.shape)
    distances = np.sum(distance_vectors * distance_vectors, axis=1)
    distance_index_tuples = list(zip(distances, range(len(distances))))
    distance_index_tuples.sort()
    print(distances.shape)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()