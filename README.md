# MotifConv
This is the official implementation of [Motif-based Graph Representation Learning with Application to Chemical Molecules](https://www.mdpi.com/2227-9709/10/1/8).
![Motif Convotion Module](https://github.com/yifeiwang15/MotifConv/blob/main/pipeline.png)

## Abstract
This work considers the task of representation learning on the attributed relational graph (ARG). 
Both the nodes and edges in an ARG are associated with attributes/features allowing ARGs to encode rich structural information 
widely observed in real applications. Existing graph neural networks offer limited ability to capture complex interactions 
within local structural contexts, which hinders them from taking advantage of the expression power of ARGs. 
We propose **M**otif **C**onvolution **M**odule (MCM), a new motif-based graph representation learning technique 
to better utilize local structural information. 
The ability to handle continuous edge and node features is one of MCM's advantages over existing motif-based models. 
MCM builds a motif vocabulary in an unsupervised way and deploys a novel motif convolution operation 
to extract the local structural context of individual nodes, which is then used to learn higher-level node representations 
via multilayer perceptron and/or message passing in graph neural networks. 
When compared with other graph learning approaches to classifying synthetic graphs, 
our approach is substantially better in capturing structural context. 
We also demonstrate the performance and explainability advantages of our approach by applying it to several molecular benchmarks.  

## Modules
Folder `graph_match` contains the CUDA-enabled system for ARG matching. Folders `MCM_for_syn`, `MCM_for_molecule_benchmarks` and `MCM_for_qm9` include all scripts to run experiments.

## Citation

If you use the codes, please cite the following paper:

```
@Article{informatics10010008,
AUTHOR = {Wang, Yifei and Chen, Shiyang and Chen, Guobin and Shurberg, Ethan and Liu, Hang and Hong, Pengyu},
TITLE = {Motif-Based Graph Representation Learning with Application to Chemical Molecules},
JOURNAL = {Informatics},
VOLUME = {10},
YEAR = {2023},
NUMBER = {1},
ARTICLE-NUMBER = {8},
URL = {https://www.mdpi.com/2227-9709/10/1/8},
ISSN = {2227-9709},
DOI = {10.3390/informatics10010008}
}
```
