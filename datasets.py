#!python3
import numpy as np
from tqdm import tqdm

def read_fvecs(filename):
    with open(filename, 'rb') as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if vec_size.size == 0:
                break
            vec = np.fromfile(f, dtype=np.float32, count=vec_size[0])
            yield vec


def read_ivecs(filename):
    with open(filename, 'rb') as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if vec_size.size == 0:
                break
            vec = np.fromfile(f, dtype=np.int32, count=vec_size[0])
            yield vec

def load_sift_dataset(train_file, test_file, groundtruth_file):
    train_data = None
    test_data = None
    groundtruth_data = None
    if train_file: 
        train_data = np.array(list(read_fvecs(train_file)))
    if test_file:
        test_data = np.array(list(read_fvecs(test_file)))
    if groundtruth_file:
        groundtruth_data = np.array(list(read_ivecs(groundtruth_file)))

    return train_data, test_data, groundtruth_data

def calculate_recall(hnsw, test, groundtruth=None, k=5, ef=10):
    if groundtruth is None:
        groundtruth = []
        print("Ground truth not found. Calculating ground truth...")
        for query in tqdm(test):
            groundtruth.append([idx for idx, dist in sorted(map(lambda a: (a[0], hnsw.distance_func(query, a[1])) , hnsw.data.items() ) , key=lambda a: a[1])[:k]])

    print("Calculating recall...")
    recalls = []
    total_calc = 0
    for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
        true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
        observed = [neighbor for neighbor, dist in hnsw.search(q=query, k=k, ef=ef, return_observed = True)]
        total_calc = total_calc + len(observed)
        results = observed[:k]
        intersection = len(set(true_neighbors).intersection(set(results)))
        # print(f'true_neighbors: {true_neighbors}, results: {results}. Intersection: {intersection}')
        recall = intersection / k
        recalls.append(recall)

    return float(np.mean(recalls)), total_calc/len(test)