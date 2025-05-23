# Three Algorithms for Merging Hierarchical Navigable Small World Graphs

This repository contains a Python implementation of three efficient algorithms for merging Hierarchical Navigable Small World (HNSW) graphs:

* **NGM**: Naive Graph Merge
* **IGTM**: Intra Graph Traversal Merge
* **CGTM**: Cross Graph Traversal Merge

For more details, please refer to the paper:
**"Three Algorithms for Merging Hierarchical Navigable Small World Graphs"**
Available on [arXiv:2505.16064](https://arxiv.org/abs/2505.16064)

Below are examples of the **NGM**, **IGTM** algorithms in action.
Blue points represent the vertices of the first graph, while green points represent the vertices of the second graph.
The edges of the first and second graphs are omitted for clarity.
The edges of the merged graph are shown as black lines.
Black points indicate the vertices whose neighborhoods in the merged graph were formed.
Yellow points represent the set of vertices for which the distances were calculated.
The red point is the vertex ![v-star](https://latex.codecogs.com/svg.image?v%5E%7B%2A%7D ) for which neighborhood is forming.

## Naive Graph  Merge (NGM)

![NGM example](animations/NGM-n1000k5-small2.gif)
<br>
NGM algorithm is a straightforward method for merging. It don't take into account information about closeness of object in all graph.
To obtain set of candidates it uses standard HNSW-Search function. The vertex for which neighborhood is forming, is selected 
in arbitrary maneer, whithout taking acount inforamtion of the previous steps.

## Itra Graph Traversal Merge (IGTM)

![IGTM example](animations/IGTM-n1000k5-small2.gif)
<br>
The most effort of the NGM algorithm lies in obtaining the set of neighborhood candidates from the other graph utilizing the HNSW-Search procedure, which every time traverses the layer graphs from the top level down to the layer number $L$. The number of computations can be reduced if we select the next vertex to process ![v-star](https://latex.codecogs.com/svg.image?v%5E%7B%2A%7D ) close to the previous one, instead of randomly choosing it. Thus, for the new ![v-star](https://latex.codecogs.com/svg.image?v%5E%7B%2A%7D ) the neighborhood candidates will also be close to the previous candidates set. To search for these new neighborhood candidates we can use the LocalSearch procedure which traverses the same graph staring from the previous neighborhood candidates set.

## Cross Graph Traversal Merge (CGTM)
CGTM is similar to IGTM, but the next vetrex ![v-star](https://latex.codecogs.com/svg.image?v%5E%7B%2A%7D ) whose neighborhood is being formed, is selected from both graphs.


## Citation

If you use this work, please cite the following paper:

```bibtex
@misc{ponomarenko2025algorithmsmerginghierarchicalnavigable,
  title        = {Three Algorithms for Merging Hierarchical Navigable Small World Graphs},
  author       = {Alexander Ponomarenko},
  year         = {2025},
  eprint       = {2505.16064},
  archivePrefix= {arXiv},
  primaryClass = {cs.DS},
  url          = {https://arxiv.org/abs/2505.16064}
}
```