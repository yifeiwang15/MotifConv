from __future__ import division
from __future__ import print_function

import os.path as osp
import time
import datetime
import os
import math
import logging
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from model import MXMNetMCM, Config
from utils import EMA
from qm9_dataset import QM9

parser = argparse.ArgumentParser()
parser.add_argument('--extend-file', type=str, help='Path for the file with node features learned by MCM.')
parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
parser.add_argument('--seed', type=int, default=920, help='Random seed.')
parser.add_argument('--epochs', type=int, default=900, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
parser.add_argument('--dataset', type=str, default="QM9", help='Dataset')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--target', type=int, default="0", help='Index of target (0~11) for prediction')
parser.add_argument('--cutoff', type=float, default=5.0, help='Distance cutoff used in the global layer')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

target = args.target
if target in [7, 8, 9, 10]:
    target = target + 5

set_seed(args.seed)

targets = ['mu (D)', 'a (a^3_0)', 'e_HOMO (eV)', 'e_LUMO (eV)', 'delta e (eV)', 'R^2 (a^2_0)', 'ZPVE (eV)', 'U_0 (eV)', 'U (eV)', 'H (eV)', 'G (eV)', 'c_v (cal/mol.K)', ]
#Change the unit from meV to eV for energy-related targets

def test(loader):
    error = 0
    ema.assign(model)

    for data in loader:
        data = data.to(device)
        output = model(data)
        error += (output - data.y).abs().sum().item()
    ema.resume(model)
    return error / len(loader.dataset)

class MyTransform(object):
    def __init__(self, add_feats=None, dic=None):
        self.target = target
        self.add_feats = add_feats
        self.dic = dic
            
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, self.target]
        #data.x = data.x[:, [0,1,2,3,4,6,7,8,9,10]]
        if self.add_feats is not None:
            size = self.add_feats[0][1].shape[1] # num of add feats
            if dic is None:
                idx = data.idx.item()
            else:
                idx = dic[data.idx.item()]
            
            if idx < len(self.add_feats):
                tmp = self.add_feats[idx][1]
                assert tmp.shape[0] == data.x.shape[0]
            else:
                tmp = torch.zeros([data.x.shape[0], size])
            data.add_feats = tmp
        
        return data

def get_savedir(target, dataset):
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(
        os.environ["LOG_DIR"], date, target, dataset + "_logs" + dt.strftime('_%H_%M_%S')
    )
    os.makedirs(save_dir)
    return save_dir

#Download and preprocess dataset

# load qm9 data withno transformation
path = osp.join(os.environ["DATA_PATH"], 'data', 'QM9')
dataset = QM9(path)
print('# of graphs:', len(dataset))

# load extra node features. 
extra_feats = torch.load(args.extend_file)

idx = []
for i in range(len(dataset)):
    idx.append(dataset[i].idx.item())

dic = dict([(idx_, i) for i, idx_ in enumerate(idx)])


# load qm9 data with transformation
transform = T.Compose([MyTransform(extra_feats, dic)])
dataset = QM9(path, transform=transform).shuffle()

save_dir = get_savedir(str(args.target), 'QM9')


# Split datasets.
# Use the same random seed as the official DimeNet` implementation.
# random_state = np.random.RandomState(seed=42)
# perm = torch.from_numpy(random_state.permutation(np.arange(130462)))
# train_idx = perm[:110000]
# val_idx = perm[110000:120000]
# test_idx = perm[120000:]
# train_dataset, val_dataset, test_dataset = dataset[train_idx], dataset[val_idx], dataset[test_idx]

train_dataset = dataset[:110000]
val_dataset = dataset[110000:120000]
test_dataset = dataset[120000:]

#Load dataset
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

print('Loaded the QM9 dataset. Target property: ', targets[args.target])

# create logger
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=os.path.join(save_dir, "train.log")
)

# stdout logger
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logging.info("Saving logs in: {}".format(save_dir))

logging.info("Task: " + str(args.target) + "\n")

# Load model
config = Config(dim=args.dim, n_layer=args.n_layer, cutoff=args.cutoff, extend_dim=dataset[0].add_feats.shape[1])

model = MXMNetMCM(config).to(device)
logging.info('Loaded the MXMNetMCM.')

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

ema = EMA(model, decay=0.999)

logging.info('===================================================================================')
logging.info('                                Start training:')
logging.info('===================================================================================')

best_epoch = None
best_val_loss = None

for epoch in range(args.epochs):
    loss_all = 0
    step = 0
    model.train()

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = F.l1_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
        optimizer.step()
        
        curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
        scheduler_warmup.step(curr_epoch)

        ema(model)
        step += 1

    train_loss = loss_all / len(train_loader.dataset)

    val_loss = test(val_loader)

    if best_val_loss is None or val_loss <= best_val_loss:
        test_loss = test(test_loader)
        best_epoch = epoch
        best_val_loss = val_loss
        logging.info("Update the best model.")

    logging.info('Epoch: {:03d}, Train MAE: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch+1, train_loss, val_loss, test_loss))

logging.info('===================================================================================')
logging.info('Best Epoch:', best_epoch)
logging.info('Best Test MAE:', test_loss)
