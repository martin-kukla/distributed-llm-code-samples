# This is a toy example how to distribute Transformer's FFN sublocks among many GPUs
# It shows how to implmement DDP and FSDP almost from the first principle:
# Initially, I meant to use torch.cuda.nccl directly, but there is a known issue (https://github.com/pytorch/pytorch/issues/38019).
# Thus, using torch.distributed's init_process_group + its communication collectives.
#
# To test, run python distributed-ffn.py -i 16 -bs 8192 -d 8192
#
# Remaining TODO: overlap communication with computation 

import argparse
import math
import os
import torch
import torch.distributed as dist # we will only use init_process_group and communication collectives (all_reduce etc.)
#import torch.cuda.nccl as nccl
import multiprocessing as mp
import time

nGPUs = torch.cuda.device_count()
if nGPUs==1:
    raise Exception("Only 1GPU available")

LR = 0.00001 # 0.1 for testing


### PARAMS + MODEL

def init_linear_layer(m, n, scale=2e-2): 
    return scale * torch.randn((n, m))
    
def init_tlayer_ffn(emb_dim, ffn_dim):
    return [init_linear_layer(emb_dim, ffn_dim)] +  [init_linear_layer(ffn_dim, emb_dim)]

def linear_fwd(layer_params, x): # input: seq_len x emb_dim
    return torch.matmul(x, torch.transpose(layer_params, 0, 1))

def t_linear_bkwd(dloss_dx, layer_params, x): # input: N x D
    return torch.einsum('bc, bd -> cd', dloss_dx, x), torch.einsum('bc, cd -> bd', dloss_dx, layer_params)
    

def tlayer_ffn_fwd(layer_params, x): # input: seq_len x emb_dim
    x = linear_fwd(layer_params[0], x)
    x = linear_fwd(layer_params[1], x)
    return x


def tlayer_ffn_bkwd(dloss_dx, layer_params, x):
    x_in = x
    x = linear_fwd(layer_params[0], x)
    
    # propagate back
    ffn2_dloss_dp, dloss_dx = t_linear_bkwd(dloss_dx, layer_params[1], x)
    ffn1_dloss_dp, dloss_dx = t_linear_bkwd(dloss_dx, layer_params[0], x_in)

    return dloss_dx.reshape(x_in.shape), (ffn1_dloss_dp, ffn2_dloss_dp)

#### Training methods: 1GPU, DDP, FSDP

# 1 GPU

def train_1gpu(layer_params, seeds, batch_size, steps=2):
    gen=torch.Generator()
    model_size = layer_params[0].shape[1]
    
    layer_params = [p.cuda(0) for p in layer_params]

    for seed in seeds.numpy().tolist():
        # Get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(0)
        dloss_dx = torch.randn((batch_size, model_size), generator=gen).cuda(0)
        
        # Forward
        y = tlayer_ffn_fwd(layer_params, x)
        
        # Backward
        _, dloss_dp = tlayer_ffn_bkwd(dloss_dx, layer_params, x)       
        
        # Optimizer step (just SGD for now)
        layer_params = [p-LR*g for p, g in zip(layer_params, dloss_dp)]
    
    return layer_params

# Multi-GPU
def init_process(rank, layer_params, seeds, batch_size, fn):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=nGPUs)
    
    fn(rank, layer_params, seeds, batch_size)
    
# DDP
def train_process_ddp(local_rank, layer_params, seeds, batch_size):
    # Probably we don't need to specify it explicitly, as the default group will do. TODO: confirm
    #group = dist.new_group(range(nGPUs))
    gen=torch.Generator()
    model_size = layer_params[0].shape[1]

    for seed in seeds.numpy().tolist():
        # get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)
        dloss_dx = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)

        # Forward
        y = tlayer_ffn_fwd(layer_params, x)

        # Backward
        dloss_dp = tlayer_ffn_bkwd(dloss_dx, layer_params, x)[1]
        dist.all_reduce(dloss_dp[0], op=dist.ReduceOp.SUM) #, group=group)       
        dist.all_reduce(dloss_dp[1], op=dist.ReduceOp.SUM) #, group=group)
        
        # Optimzer (in place)
        for param, grad in zip(layer_params, (dloss_dp[0], dloss_dp[1])):
            param.add_(-LR*grad)
  
def train_ddp(layer_params, seeds, batch_size):
    assert len(seeds) % nGPUs == 0

    def clone_layer_params(layer_params, device):
        return tuple([torch.clone(p).cuda(device) for p in layer_params])
    gpus_layer_params = [clone_layer_params(layer_params, i) for i in range(nGPUs)] 
    cpus_seeds = [t.reshape(-1) for t in seeds.reshape((-1, nGPUs)).chunk(nGPUs, dim=1)]

    processes = []
    for rank in range(nGPUs):
        p = mp.Process(target=init_process, args=(rank, gpus_layer_params[rank], cpus_seeds[rank], batch_size, train_process_ddp))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return gpus_layer_params[0]

# FSDP
def train_process_fsdp(local_rank, layer_params, seeds, batch_size):
    # Probably we don't need to specify it explicitly, as the default group will do. TODO: confirm
    #group = dist.new_group(range(nGPUs))
    gen=torch.Generator()

    sharded_p0 = layer_params[0]
    sharded_p0s = [torch.zeros_like(sharded_p0).cuda(local_rank) for _ in range(nGPUs)]
    dist.all_gather(sharded_p0s, sharded_p0) 
    sharded_p1 = layer_params[1]
    sharded_p1s = [torch.zeros_like(sharded_p1).cuda(local_rank) for _ in range(nGPUs)]
    dist.all_gather(sharded_p1s, sharded_p1)
    
    layer_params = (torch.cat(sharded_p0s), torch.cat(sharded_p1s, dim=1))
    #if local_rank==0:
    #    print(f'local_rank {local_rank}', layer_params[0].shape, layer_params[1].shape)
    #    print(f'local_rank {local_rank}', layer_params[0], layer_params[1])
    model_size = layer_params[0].shape[1]

    for seed in seeds.numpy().tolist():
        # get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)
        dloss_dx = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)

        # Forward
        y = tlayer_ffn_fwd(layer_params, x)

        # Backward
        dloss_dp = tlayer_ffn_bkwd(dloss_dx, layer_params, x)[1]
        dist.all_reduce(dloss_dp[0], op=dist.ReduceOp.SUM) #, group=group)       
        dist.all_reduce(dloss_dp[1], op=dist.ReduceOp.SUM) #, group=group)
        
        # Optimzer (in place)
        for param, grad in zip(layer_params, (dloss_dp[0], dloss_dp[1])):
            param.add_(-LR*grad)
  
def train_fsdp(layer_params, seeds, batch_size):
    assert len(seeds) % nGPUs == 0

    def chunk_p(p, dim):
       return [p_chunk.cuda(i) for i, p_chunk in enumerate(p.chunk(nGPUs, dim=dim))]
    chunked_params = [chunk_p(p, dim=i%2) for i, p in enumerate(layer_params)]
    gpus_layer_params = list(map(list, zip(*chunked_params)))
    cpus_seeds = [t.reshape(-1) for t in seeds.reshape((-1, nGPUs)).chunk(nGPUs, dim=1)]

    processes = []
    for rank in range(nGPUs):
        p = mp.Process(target=init_process, args=(rank, gpus_layer_params[rank], cpus_seeds[rank], batch_size, train_process_fsdp))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    return (torch.cat([gpu_p[0].cuda(0) for gpu_p in gpus_layer_params]), torch.cat([gpu_p[1].cuda(0) for gpu_p in gpus_layer_params], dim=1))
    
#### Setup:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iters', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-d', '--model_size', type=int, default=4)
    parser.add_argument('-m', '--mode', type=int, default=0)
    args = parser.parse_args()
    
    print(f'ARGS:\n iters:{args.iters}\n BS:{args.batch_size}\n D:{args.model_size}\n')

    seeds = torch.randint(100_000, (args.iters,)) # mock for dataset
    layer_params = init_tlayer_ffn(args.model_size, 4*args.model_size)

    num_params = sum([p.numel() for p in layer_params])
    def _gb(t_numel):
        return 4*t_numel/(1024 * 1024 * 1024)
    print(f'PARAMS: {num_params:_} (size {_gb(num_params)} GB)')
    print(f'\n')

    print(f'initial layer_params', layer_params[0].shape, layer_params[1].shape)
    print(f'initial layer_params', layer_params[0][:5,:5], layer_params[1][:5,:5])
    
    fns = [train_1gpu, train_ddp, train_fsdp]
    mp.set_start_method('spawn')
    for i, fn in enumerate(fns):
        if args.mode==0 or args.mode==i+1:
            t0 = time.time()
            fn_layer_params = fn(layer_params, seeds, args.batch_size)
            t1 = time.time()
            print(f'{fn.__name__} takes {t1-t0} seconds: ', fn_layer_params[0].shape, fn_layer_params[0][:5,:5], fn_layer_params[1].shape, fn_layer_params[1][:5,:5])

    # if args.mode==0:
    #     assert torch.allclose(n_layer_params_ddp[0], n_layer_params_1gpu[0]), f"n_layer_params_ddp[0] {n_layer_params_ddp[0]} n_layer_params_1gpu[0] {n_layer_params_1gpu[0]}"
    #     assert torch.allclose(n_layer_params_ddp[1], n_layer_params_1gpu[1]), f"n_layer_params_ddp[1] {n_layer_params_ddp[1]} n_layer_params_1gpu[1] {n_layer_params_1gpu[1]}"
