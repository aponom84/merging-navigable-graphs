#!python
# -*- coding: utf-8 -*-

from hnsw import HNSW
from hnsw import heuristic
import numpy as np
from datasets import load_sift_dataset, calculate_recall
from merge_hnsw import IGTM
import os.path
import pandas as pd 

distance_count = 0
def l2_distance(a, b):
    global distance_count
    distance_count+=1
    return np.linalg.norm(a - b)

result_file = 'results/IGTM_result.csv'

k=5
efs=[32,40,50,64,72]


jump_ef=20
local_ef=5
next_step_k=5
next_step_ef=3
M = 5

merge_params_list = [      
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 1},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 2},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 3},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 20},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 1},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 2},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 3},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 20},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 1},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 2},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 3},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 5},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':3, 'M': 10},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':3, 'M': 1},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':3, 'M': 2},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':3, 'M': 3},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':3, 'M': 5},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':3, 'M': 10},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':10, 'M': 1},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':10, 'M': 2},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':10, 'M': 3},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':10, 'M': 5},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'next_step_ef':10, 'M': 10},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':10, 'M': 1},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':10, 'M': 2},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':10, 'M': 3},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':10, 'M': 5},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'next_step_ef':10, 'M': 10}
]

hnsw_a = HNSW( distance_func=l2_distance, m=16, m0=32, ef=32, ef_construction=32,  neighborhood_construction = heuristic)
hnsw_b = HNSW( distance_func=l2_distance, m=16, m0=32, ef=32, ef_construction=32,  neighborhood_construction = heuristic)

print('Loaing hnsw_a')
hnsw_a.load('../save/sift1m/hnsw_a.txt')
print('Loaing hnsw_b')
hnsw_b.load('../save/sift1m/hnsw_b.txt')

merged_data = hnsw_a.data.copy()
merged_data.update(hnsw_b.data)

_, test_data, groundtruth_data = load_sift_dataset(train_file = None,
                                                      test_file='../datasets/sift1m-128d/sift_query.fvecs',
                                                      groundtruth_file='../datasets/sift1m-128d/sift_groundtruth.ivecs')

for merge_params in merge_params_list:
    exp = {'algorithm': 'IGTM'}
    exp['params'] = merge_params
    print('Executing:', merge_params)

    distance_count = 0
    hnsw_merged1 = IGTM(hnsw_a, hnsw_b, merged_data, 
                          jump_ef=merge_params['jump_ef'], 
                          local_ef=merge_params['local_ef'], 
                          next_step_k=merge_params['next_step_k'],
                          next_step_ef=merge_params['next_step_ef'],
                          M=merge_params['M'])
    exp['merge distance count'] = distance_count
    print('merge distance count', distance_count)

    # print('saving to disk')
    # hnsw_merged1.save(f"../save/sift1m/hnsw_merged2_jef{merge_params['jump_ef']}_lef{merge_params['local_ef']}_nsk{merge_params['next_step_k']}_M{merge_params['M']}.txt")

    for ef in efs:
        distance_count = 0
        recall, _ = calculate_recall(hnsw_merged1, test_data, groundtruth=groundtruth_data, k=5, ef=ef)
        exp[f'ef={ef} {k}@recall'] = recall
        exp[f'ef={ef} dist count'] = distance_count/len(test_data)
        print(f'ef={ef} recall: {recall}, avg dist: {distance_count/len(test_data) }') 

    df = pd.DataFrame([exp])
    if os.path.isfile(result_file):
        df.to_csv(result_file, mode='a', index=False, header=False)
    else:
        df.to_csv(result_file, mode='w', index=False, header=True)