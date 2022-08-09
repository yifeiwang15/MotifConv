# Implementation of MCM-LR for synthetic data as well as GNN baselines.

# Requirements
```
python 3.9
pytorch 1.10.0
numpy 1.21.2
dgl 0.7.2
sklearn 1.0
```



   
# Datasets
We provide a sample `syn-data-500.pt` dataset of size 500 as an example. 

We provide a sample `extended_syn-data-500.pt` including node feature learned by motif convolution.

See `syn_data.py` to generate dataset of any size.

# How to run
To reproduce the work of MXM_LR, run `mcm-main.py`.

To reproduce the work from GNNs, run

```
python main.py --model GCN --dataset 500 --batchsize 32 \
               --readout max --epoch 500 --lr 1e-3  \
               --edge True --hidden 32
```
Optional arguments:
```
    --model          {GIN, GAT, GCN} 
                     The convolution model.
    --dataset        500 10000
                     Dataset size. 
    --batchsize      batch size
    --readout        {mean, max}
                     global pooling
    --epoch          maximal training epochs
    --lr             leanring rate
    --edge           {True, False} 
                     whether to take edge weight normalization in conv.
    --hidden         hidden dimension
```