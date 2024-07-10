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

    kmeans = KMeans()
    kmeans.add_points(points[:1000])
    kmeans.fit()

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()