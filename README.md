# simpleVectorANN
A simple implementation of a vector database with approximate nearest-neighbor search using k-means and inverted indices

# Usage
```
pip install -r requirements.txt
python3 -m simpleVectorANN.main \
    -K <number of clusters> \
    -P <number of centroids to search> \
    -t <convergence threshold for k-means>
```