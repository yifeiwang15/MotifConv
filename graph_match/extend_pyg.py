from math import ceil
import torch
from Graph import Graph
from preprocess import load
import sys
import cuda.kernels as kernels
import torch.multiprocessing as mp
import KHopChemMotif
import time
import torch_geometric
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T

torch.multiprocessing.set_sharing_strategy('file_system')
alpha = 0.7
beta_0 = 1
beta_f = 30
beta_r = 1.075
I_0 = 50

nGPU = torch.cuda.device_count()


def toGraph(data_list):
    return [Graph(*data) for data in data_list]


def k_hop_subgraph(g, k_hop=1):
    sub_graphs = []
    for i in range(g[0].shape[0]):
        subset, sub_edge_index, _, mask, _ = KHopChemMotif.k_hop_subgraph(
            i, k_hop, g[2], relabel_nodes=True)
        sub_graphs.append(Graph(g[0][subset], g[1][mask], sub_edge_index))
    return sub_graphs


def run(graphs, motif_id, motif_graph):
    sub_graph_ptr = [0]  # pointer to the start of each graph's subgraphs
    sub_graphs = []
    for id, g in graphs:
        sub_graph_ptr.append(sub_graph_ptr[-1] + g[0].shape[0])
        sub_graphs += k_hop_subgraph(g, k_hop=1)
    M_N_A = [g.M_n for g in sub_graphs]
    M_N_B = [g.M_n for g in motif_graph]
    M_E_A = [g.M_e for g in sub_graphs]
    M_E_B = [g.M_e for g in motif_graph]
    E_A = [g.E for g in sub_graphs]
    E_B = [g.E for g in motif_graph]
    mask = torch.ones((len(sub_graphs), len(motif_graph)),
                      dtype=torch.int64, device='cuda')
    S = kernels.nodeSim(M_N_A, M_N_B, mask)  # get node similarity
    M = [s.clone() for s in S]  # init M
    M_new = kernels.updateM(M, (M_E_A, M_E_B), (E_A, E_B), S, mask, alpha, beta_0,
                            beta_f, beta_r, I_0)
    # non-slack result
    M_noslack = [m[1:, 1:] for m in M_new]
    S_noslack = [s[1:, 1:] for s in S]
    kernels.toHardAssign(M_noslack)  # update in-place
    score = kernels.matchScore(M_noslack, (M_E_A, M_E_B),
                               (E_A, E_B), S_noslack, mask, alpha).cpu()
    res = []
    for i, (id, g) in enumerate(graphs):
        s = score[sub_graph_ptr[i]:sub_graph_ptr[i + 1]]
        assert s.shape[0] == g[0].shape[0]
        #x = torch.cat((g[0], s), dim=1)
        x = s
        res.append((id, x))
    return res

class Dataloader():
    def __init__(self, batch_size, path) -> None:
        self.transform = T.Compose([T.Distance(norm=False)])
        self.QM9 = QM9(path, transform=self.transform)
        self.data = [(i, (g.x, g.edge_attr, g.edge_index)) for i, g in enumerate(self.QM9)]
        self.batch_size = batch_size
        self.num_batch = ceil(len(self.data) // batch_size)

    def __getitem__(self, index):  # return a batch
        return self.data[index * self.batch_size:(index + 1) * self.batch_size]

def proc(queue_out, motif_file, qm9_path, batch_size, pid, gid, nproc):
    try:
        torch.cuda.set_device(gid)
        motif_data = load(motif_file, True)  # load motifs
        motif_id, motif_graphs = tuple(zip(*motif_data))
        motif_graphs = toGraph(motif_graphs)
        dataloader = Dataloader(batch_size, qm9_path)
        for i in range(pid, dataloader.num_batch, nproc):
            batch = dataloader[i]
            res = run(batch, motif_id, motif_graphs)
            queue_out.put(res)
        print(f"pid {pid} on GPU {gid} finished")
    except Exception as e:
        print(f"pid {pid} on GPU {gid} failed: {e}")
        raise e
    finally:
        queue_out.put(None)


def aggregate(queue, num_proc):
    # aggregate the results
    # the queue is a list of tuples (id1, id2, score)
    res = []
    count = 0
    start = time.time()
    while count < num_proc:
        item = queue.get()
        if item is None:
            count += 1
        else:
            res += item
            if len(res) % 2 == 1:
                end = time.time()
                with open("log.txt", 'w') as f:
                    f.write("%.3fs: %d\n"%((end-start), len(res)))
    return res

def main():
    mp.set_start_method('spawn')
    m = mp.Manager()
    qm9_path = sys.argv[1]
    motif_file = sys.argv[2]
    batch_size = int(sys.argv[3])
    num_proc = nGPU * 2
    queue_out = m.Queue(maxsize=num_proc)
    pool = mp.Pool(num_proc)
    status = []
    start = time.time()
    
    for pid in range(num_proc):
        gid = pid % nGPU
        result = pool.apply_async(proc, args=(queue_out, motif_file, qm9_path, batch_size,
                                              pid, gid, num_proc))
        status.append(result)
    pool.close()
    res = aggregate(queue_out, num_proc)
    pool.join()
    #res = sorted(res, key=lambda x: x[0])
    outname = motif_file.split('/')[-1].split('.')[0]
    torch.save(res, f'extended_{outname}.pt')
    try:
        for s in status:
            s.get()
    except Exception as e:
        print(e.with_traceback())
        pool.terminate()
    else:
        end = time.time()
        print('time: ', end - start)


if __name__ == '__main__':
    main()
