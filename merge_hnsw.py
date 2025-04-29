#!python
# -*- coding: utf-8 -*-
from copy import deepcopy
from tqdm import tqdm
from hnsw import HNSW
import random

# Возможно стоит назвать naive_merge или default_merge
def insertion_merge(hnsw_a, hnsw_b, ef_construction):
    '''
    Insert nodes from hnsw_b to hnsw_a by simple insertion

    hnsw_a    – the first hnsw graph 
    hnsw_b    – the second hnsw graph
    ef_construction – ef parameter for searching candidates in the second graph
    '''

    hnsw_c = deepcopy(hnsw_a)
    hnsw_c.ef_construction = ef_construction
    # hnsw_c = HNSW(distance_func=hnsw_a.distance_func, m=hnsw_a.m, m0=hnsw_a.m0, ef=hnsw_a.ef, ef_construction=hnsw_a.ef_construction, neighborhood_construction = hnsw_a.neighborhood_construction) 
    for key, point in tqdm(hnsw_b.data.items()):
        hnsw_c.add(key, point)
    return hnsw_c


def hnsw_general_merge(hnsw_a, hnsw_b, merged_data, layer_merge_func):
    hnsw_merged = HNSW(distance_func=hnsw_a.distance_func, m=hnsw_a.m, m0=hnsw_a.m0, ef=hnsw_a.ef, ef_construction=hnsw_a.ef_construction, neighborhood_construction = hnsw_a.neighborhood_construction)
    hnsw_merged.data = merged_data 
    hnsw_merged.graphs = [] # sequence of merged graphs (layers) denoted as $G^C_i$
    levels_merged_max = max(len(hnsw_a.graphs), len(hnsw_b.graphs))
    levels_merged_min = min(len(hnsw_a.graphs), len(hnsw_b.graphs))
    
    for level in range(levels_merged_min): 
        print('Merging level:', level)
        hnsw_merged.graphs.append(  layer_merge_func(hnsw_a, hnsw_b, merged_data, level) )

    for level in range(levels_merged_min, levels_merged_max):
        if len(hnsw_a.graphs) >= len(hnsw_b.graphs):
            hnsw_merged.graphs.append(hnsw_a.graphs[level])
        else:
            hnsw_merged.graphs.append(hnsw_b.graphs[level])

    if len(hnsw_a.graphs) >= len(hnsw_b.graphs):
        hnsw_merged.enter_point = hnsw_a.enter_point
    else:
        hnsw_merged.enter_point = hnsw_b.enter_point

    return hnsw_merged

# TODO переименовать в simple_merge
def merge_naive_layer(hnsw_a, hnsw_b, merged_data, level, search_ef=5):
    '''
    hnsw_a    – the first hnsw graph 
    hnsw_b    – the second hnsw graph
    level     – mering level number
    search_ef – ef parameter for searching candidates in the second graph                  
    '''
    m = hnsw_a.m0 if level == 0 else hnsw_a.m
    merged_edges = {}
    for curr_idx in tqdm(hnsw_a.graphs[level].keys()): 
        candidates_b = hnsw_b.search(q=hnsw_a.data[curr_idx], k=m, ef=search_ef, level=level, return_observed=False) 
        # == build neighborhood for curr_idx and save to externalset of edges  ==
        candidates = [ (idx_b, dist) for idx_b, dist in candidates_b] + [ (idx, dist) for idx, dist in hnsw_a.graphs[level][curr_idx]]
        # merged_edges[curr_idx] = sorted ([ (idx_b + len(kga.data), dist) for idx_b, dist in candidates_b] + [ (idx, dist) for idx, dist in kga.edges[curr_idx]], key=lambda a: a[1])[:k]
        merged_edges[curr_idx] = hnsw_a.neighborhood_construction(candidates, hnsw_a.data[curr_idx], m, hnsw_a.distance_func, merged_data)    
        # == == == == == == == == == == == == == == == == == == == == == == == ==

    for curr_idx in tqdm(hnsw_b.graphs[level].keys()): 
        candidates_a=hnsw_a.search(q=hnsw_b.data[curr_idx], k=m, ef=search_ef, level=level, return_observed=True)
        # == build neighborhood for curr_idx and save to externalset of edges  ==
        candidates = [(idx_a, dist) for idx_a, dist in candidates_a] + [(idx, dist) for idx, dist in hnsw_b.graphs[level][curr_idx]]
        merged_edges[curr_idx] = hnsw_b.neighborhood_construction(candidates, hnsw_b.data[curr_idx], m, hnsw_a.distance_func, merged_data)
        # == == == == == == == == == == == == == == == == == == == == == == == ==

    return merged_edges


def merge_naive(hnsw_a, hnsw_b, merged_data, merge_ef = 20):
    def layer_merge_naive_func(hnsw_a, hnsw_b, merged_data, level):
        return merge_naive_layer(hnsw_a, hnsw_b, merged_data, level, search_ef=merge_ef)
    return hnsw_general_merge(hnsw_a, hnsw_b, merged_data, layer_merge_naive_func)


def IGTM_layer(hnsw_a, hnsw_b, merged_data, level, jump_ef, local_ef, next_step_k, next_step_ef, M):
    '''
    hnsw_a       – first hnsw graph 
    hnsw_b       – second hnsw graph
    merged_data  – joined vectors data from the first and the second graph 
    jump_ef      - ef parameter for search starting from the top level
    local_ef     - ef parameter for search starting from some neighbours 
    next_step_k  - at each iteration we look for the next element around the current vertex in the first graph. 
                   However it can be surrounded by the "done" vertex, so we have to walk away. 
                   Thus this parameter controls how far from the current vertex we can go.
    next_step_ef – a purpose of this parameter is similar {next_step_k}
    M            – number of starting point for the local search.  
    '''
    merged_edges = {} # merged graph
    not_done = set(hnsw_a.graphs[level].keys())
    m = hnsw_a.m0 if level == 0 else hnsw_a.m
    
    # tqdm progress bar based on the initial size of the `not_done` set
    progress_bar = tqdm(total=len(not_done), desc="Merging progress")

    while not_done:
        # Start with a vertex from `not_done`
        curr_idx = not_done.pop()

        # Perform jump search on graph B
        staring_points = hnsw_b.search(q=hnsw_a.data[curr_idx], k=M, ef=jump_ef, level=level, return_observed=False)
        
        while True:
            # Perform local search at graph B
            candidates_b = hnsw_b.beam_search(graph=hnsw_b.graphs[level], q=hnsw_a.data[curr_idx], k=m, eps=staring_points, ef=local_ef, return_observed=False)
            
            # Build neighborhood for curr_idx and save to external set of edges
            candidates = [(idx_b, dist) for idx_b, dist in candidates_b] + [(idx, dist) for idx, dist in hnsw_a.graphs[level][curr_idx]]
            merged_edges[curr_idx] = hnsw_a.neighborhood_construction(candidates, merged_data[curr_idx], m, hnsw_a.distance_func, merged_data)

            # Determine new set of entry points for search in hnsw_b
            # staring_points = [idx for idx, dist in candidates_b[:m]]
            staring_points = [idx for idx, dist in candidates_b[:M]]

            # Perform local search at graph A to find next candidate
            candidates_a = hnsw_a.beam_search(graph=hnsw_a.graphs[level], q=hnsw_a.data[curr_idx], k=next_step_k, eps=[curr_idx], ef=next_step_ef, return_observed=False)
            candidates_a = [c[0] for c in candidates_a[:next_step_k] if c[0] in not_done]

            if not candidates_a:
                break

            # Move to the next candidate and remove it from `not_done`
            curr_idx = candidates_a[0]
            not_done.remove(curr_idx)
            progress_bar.update(1)

        # Update the progress bar
        progress_bar.update(1)

    progress_bar.close()
    return merged_edges


def IGTM(hnsw_a, hnsw_b, merged_data, jump_ef=20, local_ef=5, next_step_k=5, next_step_ef=3, M = 5):
    def layer_IGTM_func(hnsw_a, hnsw_b, merged_data, level) :
        merged_edges = {} 
        # phase 1) 
        merged_edges.update(IGTM_layer(hnsw_a, hnsw_b, merged_data, level=level, jump_ef=jump_ef, local_ef=local_ef, next_step_k=next_step_k, next_step_ef=next_step_ef, M = M))
        # phase 2)
        merged_edges.update(IGTM_layer(hnsw_b, hnsw_a, merged_data,  level=level, jump_ef=jump_ef, local_ef=local_ef, next_step_k=next_step_k, next_step_ef=next_step_ef, M = M))
        return merged_edges

    return hnsw_general_merge(hnsw_a, hnsw_b, merged_data, layer_IGTM_func)


def CGTM_layer(hnsw_a, hnsw_b, merged_data, level, jump_ef = 20, local_ef=5, next_step_k=3, M = 3):
    '''
    hnsw_a       – First hnsw graph 
    hnsw_b       – Second hnsw graph
    jump_ef      - ef parameter for search starting from the top level
    local_ef     - ef parameter for search starting from some neighbours 
    next_step_k  - Controls how far we can go over candidate set from current vertex to get a new current. Use next_step_k=-1 to minimize number of jumps
    M            – Number of starting points for local search
    '''
    merged_edges = {}

    m = hnsw_a.m0 if level == 0 else hnsw_a.m
    
    not_done_a = set( hnsw_a.graphs[level].keys())
    not_done_b = set( hnsw_b.graphs[level].keys())
    not_done = not_done_a.union( [i  for i in not_done_b] )

    progress_bar = tqdm(total=len(not_done), desc="Merging progress")
    
    while not_done:
        curr_idx = random.choice(list(not_done))
    
        # do jump search   
        enter_points_a = hnsw_a.search(q=merged_data[curr_idx], k=M, ef=jump_ef, level=level, return_observed=False) 
        enter_points_b = hnsw_b.search(q=merged_data[curr_idx], k=M, ef=jump_ef, level=level, return_observed=False) 
        while True:
            not_done.remove(curr_idx) # remove from not_done
            progress_bar.update(1)
            # searching for a new current

            # Do local search at graph A. Decrease local_ef to traverse closer to curr vertex
            candidates_a = hnsw_a.beam_search(graph=hnsw_a.graphs[level], q=merged_data[curr_idx], k=m, eps=enter_points_a, ef=local_ef, return_observed=False) 
            # Do local search at graph B. Decrease local_ef to traverse closer to curr vertex
            candidates_b = hnsw_b.beam_search(graph=hnsw_b.graphs[level], q=merged_data[curr_idx], k=m, eps=enter_points_b, ef=local_ef, return_observed=False)
            
            # --== build neighborhood for new_curr_idx and save to externalset of edges  ==--
            if curr_idx < len(hnsw_a.data):                
                candidates  = hnsw_a.graphs[level][curr_idx] + [ (idx_b, dist) for idx_b, dist in candidates_b]
            else:                
                candidates =  candidates_a + [(idx_b, dist) for idx_b, dist in hnsw_b.graphs[level][curr_idx]]                     
            merged_edges[curr_idx] = hnsw_a.neighborhood_construction(candidates, merged_data[curr_idx], m, hnsw_a.distance_func, merged_data)                
            # --== build neighborhood for new_curr_idx and save to externalset of edges  ==--
                  
            candidates_a_not_done = [ (idx, dist) for idx, dist in candidates_a[:next_step_k] if idx in not_done]
            candidates_b_not_done = [ (idx, dist) for idx, dist in candidates_b[:next_step_k] if idx in not_done]
            
            candidates_not_done = candidates_a_not_done + candidates_b_not_done

            if not candidates_not_done: 
                break #jump to the random point

            new_curr = min(candidates_not_done, key=lambda a: a[1])
            new_curr_idx = new_curr[0]
                        
            curr_idx = new_curr_idx
            enter_points_a = [idx for idx, dist in candidates_a]
            enter_points_b = [idx for idx, dist in candidates_b]
    return merged_edges
                           

def CGTM(hnsw_a, hnsw_b, merged_data, jump_ef=20, local_ef=5, next_step_k=3, M=3):
    def layer_CGTM_func(hnsw_a, hnsw_b, merged_data, level) :
        return CGTM_layer(hnsw_a=hnsw_a, hnsw_b=hnsw_b, merged_data=merged_data, level=level, jump_ef=jump_ef, local_ef=local_ef, next_step_k=next_step_k, M=M)
    
    return hnsw_general_merge(hnsw_a, hnsw_b, merged_data, layer_CGTM_func)    
