# Implementation of MCM-GNN for molecular datasets.

## Dataset
The user should create a `data` folder and download datasets from [MoleculeNet](https://moleculenet.org/datasets-1) 
and then run `loader.py` to process and save the data. 

MCM embeddings can be calculated following the instructions in `graph_match` folder. 

## How to run
Baselines:
```
finetune.py --dataset bbbp --emb_dim 300 --num_layer 5 --split random_scaffold --gnn_type gin
```

Proposed method:
```
finetune_extend.py --dataset bbbp --emb_dim 64 --num_layer 3 --split random_scaffold --gnn_type gin 
```

## Acknowledgment
Part of code is modified from [Motif-based Graph Self-Supervised Learning for Molecular Property Prediction](https://arxiv.org/abs/2110.00987). 
