import torch
import torch
from Orange import data, distance
from Orange.clustering import hierarchical
from collections import Counter
import sys
NUM_CLUSTERS = sys.argv[1]

subgraphs = torch.load("graphs.pt") # motif list
scores = torch.load("score.pt")

size = len(subgraphs)
S = torch.zeros([size, size])

for i, score in enumerate(scores):
    S[score[0], score[1]] = abs(score[2])
    S[score[1], score[0]] = abs(score[2])


# ignore self matching
mask = torch.eye(S.shape[0], S.shape[0]).bool()
S.masked_fill_(mask, 0)

assert S.shape[0] == len(subgraphs)
print(len(subgraphs))

# Hierarchical clustering

hierar = hierarchical.HierarchicalClustering(n_clusters=NUM_CLUSTERS)
hierar.fit(S.max().item()+0.001-S)
c = Counter(hierar.labels)

centers_id = []
for lbl in range(NUM_CLUSTERS):
    indice = [idx for idx, value in enumerate(hierar.labels == lbl) if value == True]
    max_id = torch.argmax(S[indice, :][:, indice].sum(dim=0)).item()
    centers_id.append(indice[max_id])

centers = [subgraphs[i] for i in centers_id]
torch.save(centers, "motifs.pt")