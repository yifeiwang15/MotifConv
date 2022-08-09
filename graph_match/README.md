# Implementation of motif convolution

## Requirements
- CUDA 10.2 or higher
- gcc 7.5.0
- nvcc 10.2 or higher
- Pytorch 1.9.0
- Python 3.7 or higher

## CUDA kernels compliation
JIT is used to complie CUDA kernels. 
The first time running needs compliation and the resultant files are dynamic libaraies in ```CUDA/<kernel name>```  directory. 
If it takes too long to compile, please remove files except those end with ```.cpp```, ```.h```, and ```.cu```.
The compliation usually takes 3 min. 

If you meet issues with compliation, try loading the following modules.
```
module purge
module load gnu7
module load cuda/10.2
module load anaconda
export TORCH_CUDA_ARCH_LIST="7.0+PTX"
```
`TORCH_CUDA_ARCH_LIST` depends on your GPU. V100 uses 7.0, RTX2 uses 7.5.
## Input graph format. 
Each graph should be represented as a 3-tuple `(Mn, Me, E)` where `Mn` is the node feature matrix, `Me` is the edge feature matrix and 
`E` is the matrix of edge index. These corresponds to `data.x`, `data.edge_attr` and `data.edge_index` 
in [PyG library](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html).

List of graphs are save in a pt file. 

## Run motif convolution
One needs to prepare a file of motifs `motif_vocab.pt` and a file of graphs `graphs.pt`,
run `python3 extend.py motif_vocab.pt graphs.pt 10`. 
There is one output file `extend_data.pt` containing the node feature learned by motif convolution. 
One updated node feature matrix of size `[num_node, motif_vocab_size]` is assigned to each graph.

We also implement `extend_pyg.py` with a sample motif file `qm9_motifs_100.pt` for reproducing the work of MCM in QM9.
Run `python3 extend_pyg.py ./data qm9_motifs_100.pt 32`.

## Run motif vocabulary construction
Clustering implementation depends on Orange3 library. [Installation here](https://orangedatamining.com/download/#macos). 

Given a file of graphs `graphs.pt`, first compute pairwise similarity and then build the vocabulary via clustering.
Run `python3 run_gm.py graphs.pt`, which produces an output file `scores.pt`. It contains similarity for any graph pairs in `graphs.pt`.
Then obtain motif vocabulary of size n by running `python3 clusterings.py n`. The output file `motifs.pt` contains n motifs that will be further used in motif convolution.


## More on graph matching
Modify `cuda/similarity.h` if you have new node-wise/edge-wise similarity function.