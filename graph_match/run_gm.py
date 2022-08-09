import torch
from Graph import Graph
from preprocess import load, load_raw, convert_g_to_matrices
import time
import sys
from cuda.kernels import nodeSim, updateM, triu_k2ij, matchScore, toHardAssign
from math import ceil
import multiprocessing as mp



class Dataloader:
    def __init__(self, data, batch_size):
        self.idx = [d[0] for d in data]
        self.data = [d[1] for d in data]
        self.batch_size = batch_size
        self.num_batches = ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index):  # return a batch
        if index >= self.num_batches * (self.num_batches + 1) // 2:
            raise ValueError('index out of range')
        # row starts from 0, col from 1
        row, col = triu_k2ij(self.num_batches + 1, index)
        col = col - 1  # col starts from 0
        if col == row:  # the triangle workload
            idx = self.idx[row * self.batch_size: (row + 1) * self.batch_size]
            A = self.data[row * self.batch_size: (row + 1) * self.batch_size]
            return (list(zip(idx, A)),)
        else:
            idx_A = self.idx[row *
                             self.batch_size: (row + 1) * self.batch_size]
            A = self.data[row * self.batch_size: (row + 1) * self.batch_size]
            idx_B = self.idx[col *
                             self.batch_size: (col + 1) * self.batch_size]
            B = self.data[col * self.batch_size: (col + 1) * self.batch_size]
            return (list(zip(idx_A, A)), list(zip(idx_B, B)))


def toGraph(data_list):
    return [Graph(*data) for data in data_list]


def run(queues, data_list, batch_size, comm_rank, comm_size):
    queue_s = queues[0]
    dataloader = Dataloader(data_list, batch_size)
    num_workloads = (dataloader.num_batches + 1) * dataloader.num_batches // 2
    for i in range(comm_rank, num_workloads, comm_size):
        data = dataloader[i]
        if len(data) == 1:
            idx = [d[0] for d in data[0]]
            g_data = [d[1] for d in data[0]]
            g_lst = toGraph(g_data)

            mask = torch.ones((len(g_data), len(g_data)), dtype=torch.int64, device=torch.device('cuda'))
            
            M_N = [g.M_n for g in g_lst]
            M_E = [g.M_e for g in g_lst]
            E = [g.E for g in g_lst]
            S = nodeSim(M_N, mask)  # get node similarity
            M = [s.clone() for s in S]  # init M
            M_new = updateM(M, (M_E,), (E,), S, mask, alpha, beta_0,
                            beta_f, beta_r, I_0)  # start optimization
            toHardAssign(M_new)

            # non-slack result
            # M_noslack = [m[1:, 1:] for m in M_new]
            # S_noslack = [s[1:, 1:] for s in S]

            # M_new is a list of tensors
            score = matchScore(M_new, (M_E,), (E,), S, mask, alpha).abs_().cpu()
            # add scores to the queue
            for i, s in enumerate(score):
                id1, id2 = triu_k2ij(len(idx), i)
                queue_s.put((id1 + idx[0], id2 + idx[0], s.item()))
                #queue_m.put((id1 + idx[0], id2 + idx[0], M_new[i].cpu()))

        elif len(data) == 2:
            idx_A = [d[0] for d in data[0]]
            idx_B = [d[0] for d in data[1]]
            g_data_A = [d[1] for d in data[0]]
            g_data_B = [d[1] for d in data[1]]
            g_lst_A = toGraph(g_data_A)
            g_lst_B = toGraph(g_data_B)
            
            mask = torch.ones((len(g_lst_A), len(g_lst_B)), dtype=torch.int64, device=torch.device('cuda'))

            M_N_A = [g.M_n for g in g_lst_A]
            M_N_B = [g.M_n for g in g_lst_B]
            M_E_A = [g.M_e for g in g_lst_A]
            M_E_B = [g.M_e for g in g_lst_B]
            E_A = [g.E for g in g_lst_A]
            E_B = [g.E for g in g_lst_B]
            S = nodeSim(M_N_A, M_N_B, mask)  # get node similarity
            M = [s.clone() for s in S]  # init M
            M_new = updateM(M, (M_E_A, M_E_B), (E_A, E_B), S, mask, alpha, beta_0,
                            beta_f, beta_r, I_0)

            # non-slack result
            # M_noslack = [m[1:, 1:] for m in M_new]
            # S_noslack = [s[1:, 1:] for s in S]

            toHardAssign(M_new)
            score = matchScore(M_new, (M_E_A, M_E_B),
                               (E_A, E_B), S, mask, alpha).abs_().cpu()

            for r in range(score.size(0)):
                for c in range(score.size(1)):
                    if mask[r][c] == 1:
                        queue_s.put((idx_A[r], idx_B[c], score[r, c].item()))
        else:
            raise Exception('unexpected data length')


def proc(queues, filename, batch_size, pid, gid, num_proc, preprocessed=False):
    try:
        torch.cuda.set_device(gid)
        data_list = load(filename, preprocessed=preprocessed) # in matrix format
        run(queues, data_list, batch_size, pid, num_proc)
        print(f"pid {pid} on GPU {gid} finished")
    except Exception as e:
        print(f"pid {pid} on GPU {gid} failed: {e}")
        raise e
    finally:
        queues[0].put(None)
        queues[1].put(None)


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
            res.append(item)
            if len(res) % 1000 == 1:
                end = time.time()
                with open("log.txt", 'w') as f:
                    f.write("%.3fs: %d\n"%((end-start), len(res)))
    return res



alpha = 1
beta_0 = 1
beta_f = 30
beta_r = 1.075
I_0 = 50

nGPU = torch.cuda.device_count()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    m = mp.Manager()
    queue_s = m.Queue()  # queue for score
    queue_m = m.Queue()  # queue for matching matrix
    filename = sys.argv[1]
    batch_size = int(sys.argv[2])
    preprocessed = False if len(sys.argv) < 4 else bool(int(sys.argv[3]))
    num_proc = nGPU
    p = mp.Pool(num_proc)
    res = []
    with open("log.txt", 'w') as f:
        f.write("preprocessed: " + str(preprocessed) + "\n")
    start = time.time()
    for pid in range(num_proc):
        gid = pid % nGPU
        result = p.apply_async(proc, args=((queue_s, queue_m), filename, batch_size,
                                           pid, gid, num_proc, preprocessed))
        res.append(result)
    p.close()
    scores = aggregate(queue_s, num_proc)
    #Ms = aggregate(queue_m, num_proc)
    try:
        for r in res:
            r.get()
    except Exception as e:
        print(e.with_traceback())
        p.terminate()
    else:
        end = time.time()
        print('time: ', end - start)
        torch.save(scores, 'scores.pt')
        # torch.save(Ms, 'Ms.pt')
