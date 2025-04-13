"""run.py:"""
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1, 2, 3])
    for i in range(10):
        tensor = torch.ones(1).cuda(rank)
        s = torch.cuda.Stream()
        handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
        handle.wait()
        with torch.cuda.stream(s):
            # Here I could put some concurent op
            s.wait_stream(torch.cuda.default_stream())
            tensor.add_(100)
        print('Iter', i, 'Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    world_size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()