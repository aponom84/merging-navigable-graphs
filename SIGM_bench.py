#!python
# -*- coding: utf-8 -*-

from hnsw import HNSW
from hnsw import heuristic
import numpy as np
from datasets import load_sift_dataset, calculate_recall
from merge_hnsw import insertion_merge
import os.path
import pandas as pd

distance_count = 0
def l2_distance(a, b):
    global distance_count
    distance_count+=1
    return np.linalg.norm(a - b)

result_file = 'results/insertion_merge_result.csv'

k=5
efs=[32,40,50,64,72]


merge_params_list = [      
    {'ef_construction': 16},
    {'ef_construction': 24}
    # {'ef_construction': 32},
    # {'ef_construction': 40},    
]

hnsw_a = HNSW( distance_func=l2_distance, m=16, m0=32, ef=32, ef_construction=32,  neighborhood_construction = heuristic)
hnsw_b = HNSW( distance_func=l2_distance, m=16, m0=32, ef=32, ef_construction=32,  neighborhood_construction = heuristic)

print('Loading hnsw_a')
hnsw_a.load('save/sift1m/hnsw_a.txt')
print('Loading hnsw_b')
hnsw_b.load('save/sift1m/hnsw_b.txt')

merged_data = hnsw_a.data.copy()
merged_data.update(hnsw_b.data)

_, test_data, groundtruth_data = load_sift_dataset(train_file = None,
                                                      test_file='datasets/sift1m-128d/sift_query.fvecs',
                                                      groundtruth_file='datasets/sift1m-128d/sift_groundtruth.ivecs')

for merge_params in merge_params_list:
    exp = {'algorithm': 'SIGM'}
    exp['params'] = merge_params
    print('Executing:', merge_params)

    distance_count = 0
    hnsw_merged = insertion_merge(hnsw_a, hnsw_b,ef_construction = merge_params['ef_construction'])
    exp['merge distance count'] = distance_count
    print('merge distance count', distance_count)

    print('saving to disk')
    hnsw_merged.save(f"save/sift1m/hnsw_insertion_merge_ef_construction{merge_params['ef_construction'] }.txt")

    for ef in efs:
        distance_count = 0
        recall, _ = calculate_recall(hnsw_merged, test_data, groundtruth=groundtruth_data, k=5, ef=ef)
        exp[f'ef={ef} {k}@recall'] = recall
        exp[f'ef={ef} dist count'] = distance_count/len(test_data)
        print(f'ef={ef} recall: {recall}, avg dist: {distance_count/len(test_data) }') 

    df = pd.DataFrame([exp])
    if os.path.isfile(result_file):
        df.to_csv(result_file, mode='a', index=False, header=False)
    else:
        df.to_csv(result_file, mode='w', index=False, header=True)