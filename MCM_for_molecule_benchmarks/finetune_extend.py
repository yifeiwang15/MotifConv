import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model_extend import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
from warmup_scheduler import GradualWarmupScheduler
import os

import torch_geometric.transforms as T
from torch_geometric.utils import remove_isolated_nodes, contains_isolated_nodes

criterion = nn.BCEWithLogitsLoss(reduction = "none")

class MyTransform(object):
    def __init__(self, add_feats=None, dic=None):
        self.add_feats = add_feats
        self.dic = dic
        
    def __call__(self, data):
        if self.add_feats is not None:
            size = self.add_feats[0][1].shape[1] # num of add feats
            if self.dic is None:
                idx = data.id.item()
            else:
                idx = self.dic[data.id.item()]
            
            if idx < len(self.add_feats):
                tmp = self.add_feats[idx][1]
                if tmp.shape[0] != data.x.shape[0]:
                    tmp = torch.zeros([data.x.shape[0], size])
            else:
                tmp = torch.zeros([data.x.shape[0], size])
                
            data.add_feats = tmp
        
        return data
    
def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default="", help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    add_feats = torch.load("./add_feats/extended_{}_update.pt".format(args.dataset))
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    dic = dict(zip(dataset.data.id.tolist(), range(len(dataset))))
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset, transform=MyTransform(add_feats, dic))
    
    # remove incorrect graphs.
    temp = []
    for i in range(len(dataset)):
        if not contains_isolated_nodes(dataset[i].edge_index, dataset[i].x.shape[0]):
            temp.append(i)

    dataset= dataset[temp]
    print(dataset)
    
    # data augmentation
#     transform = T.Compose([MyTransform(add_feats), Interpolation(0.2)])
#     dataset_aug = MoleculeDataset("dataset/" + "bace", dataset="bace", transform=transform)
    
    
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        smiles_list = [smiles_list[i] for i in temp]
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        
        print("scaffold")
    
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        
        print("random")
        
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        smiles_list = [smiles_list[i] for i in temp]
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed = args.seed)
        
        print("random scaffold")
    
    else:
        raise ValueError("Invalid split option.")

    #print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, add_feats_dim=dataset[0].add_feats.shape[1])
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    #scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=scheduler)
    
    #print(optimizer)
    res = np.zeros([args.epochs, 3])
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)
        #scheduler.step(epoch)
        
        #print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)
        res[epoch-1][0] = train_acc
        res[epoch-1][1] = val_acc
        res[epoch-1][2] = test_acc
        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
    df = pd.DataFrame(res, columns = ['train_acc','val_acc','test_acc'])
    df.to_csv("logs_e.csv")
if __name__ == "__main__":
    main()
