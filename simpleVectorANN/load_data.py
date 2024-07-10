import os
import tarfile
from urllib.request import urlretrieve

import h5py
import numpy as np
from typing import Any


def write_output(train: np.ndarray, test: np.ndarray, groundtruth: np.ndarray, fn: str) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes 
    and stores the nearest neighbors and their distances for the test set using a 
    brute-force approach.
    
    Args:
        train (np.ndarray): The training data.
        test (np.ndarray): The testing data.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for 
            each point in the test set. Defaults to 100.
    """

    num_nearest_neighbors = len(groundtruth[0])
    with h5py.File(fn, "w") as f:
        f.attrs["dimension"] = len(train[0])
        print(f"train size: {train.shape[0]} * {train.shape[1]}")
        print(f"test size:  {test.shape[0]} * {test.shape[1]}")
        f.create_dataset("points", data=train)
        f.create_dataset("queries", data=test)
        f.create_dataset("nearest_neighbors", data=groundtruth)

def download(source_url: str, destination_path: str) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.
    
    Args:
        source_url (str): The URL of the file to download.
        destination_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(destination_path):
        print(f"downloading {source_url} -> {destination_path}...")
        urlretrieve(source_url, destination_path)
    else:
        print(f"{destination_path} already exists . . . Skipping download")


def _load_texmex_vectors(f: Any, n: int, k: int, datatype: str = "f") -> np.ndarray:
    import struct

    if datatype == "f":
        dtype = "float"
    elif datatype == "i":
        dtype = "int"
    else:
        raise TypeError(f'Expected "f" or "i" as datatype, received: {datatype}')
    v = np.zeros((n, k), dtype=dtype)
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack(datatype * k, f.read(k * 4))
        if i == 12 and datatype == "f":
            print(v[i])

    return v


def _get_irisa_matrix(t: tarfile.TarFile, fn: str, datatype: str = "f") -> np.ndarray:
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k, datatype)


def download_dataset(hdf5_filename: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = "sift.tar.tz"
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        groundtruth = _get_irisa_matrix(t, "sift/sift_groundtruth.ivecs", "i")

        print("All points: ", train.shape)
        print("All queries: ", test.shape)
        print("Ground truth: ", groundtruth.shape)
        write_output(train, test, groundtruth, hdf5_filename)