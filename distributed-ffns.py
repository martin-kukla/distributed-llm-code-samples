# This is a toy example how to distribute the computation of Transformer's FFN sublocks among GPUs
# It shows how to implmement DDP and FSDP almost from the first principle:
# Initially, I meant to use torch.cuda.nccl directly, but there is a known issue (https://github.com/pytorch/pytorch/issues/38019).
# Thus, using torch.distributed's init_process_group + its communication collectives (I believe this still being relatively a thin wrapper for NCCL)
#
# To test, run "python distributed-ffns.py --iters 16 --batch_size 8192 --model_size 8192 --method M", where M is one of:
#   "0": run all methods;  "1": run on 1GPU, "2": run DDP, "3": run FSDP (with DDP)
#
# NB: For simplicity, the random dataset is used, and no real loss function is used ( I imitate it by randomized dloss_dx coming from "right")
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

def train_1gpu(layers_params, seeds, batch_size):
    gen=torch.Generator()
    model_size = layers_params[0][0].shape[1]

    move_l = lambda l: [p.cuda(0) for p in l]
    layers_params = [move_l(l) for l in layers_params]

    for seed in seeds.numpy().tolist():
        # Get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(0)
        dloss_dx = torch.randn((batch_size, model_size), generator=gen).cuda(0)
        
        # Forward
        y=x
        acts = []
        for l in layers_params:
            acts.append(y)
            y = tlayer_ffn_fwd(l, y)
        
        # Backward + optimizer (just SGD for now)
        batch_dloss_dx = dloss_dx
        for i in reversed(range(len(layers_params))):
            batch_dloss_dx, dloss_dp = tlayer_ffn_bkwd(batch_dloss_dx, layers_params[i], acts[i])       
        
            layers_params[i] = [p-LR*g for p, g in zip(layers_params[i], dloss_dp)]
    
    return layers_params

# Multi-GPU
def init_process(rank, layers_params, seeds, batch_size, fn):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=nGPUs)
    
    fn(rank, layers_params, seeds, batch_size)
    
# DDP
def train_process_ddp(local_rank, layers_params, seeds, batch_size):
    gen=torch.Generator()
    model_size = layers_params[0][0].shape[1]

    for seed in seeds.numpy().tolist():
        # get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)
        dloss_dx = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)

        # Forward
        y=x
        acts = []
        for l in layers_params:
            acts.append(y)
            y = tlayer_ffn_fwd(l, y)

        # Backward + optimizer (in-place SGD)
        batch_dloss_dx = dloss_dx
        for i in reversed(range(len(layers_params))):
            batch_dloss_dx, dloss_dp = tlayer_ffn_bkwd(batch_dloss_dx, layers_params[i], acts[i])  
            dist.all_reduce(dloss_dp[0], op=dist.ReduceOp.SUM)    
            dist.all_reduce(dloss_dp[1], op=dist.ReduceOp.SUM)

            # Optimzer (in place)
            for param, grad in zip(layers_params[i], (dloss_dp[0], dloss_dp[1])):
                param.add_(-LR*grad)
  
def train_ddp(layers_params, seeds, batch_size):
    assert len(seeds) % nGPUs == 0

    def clone_layer_params(layer_params, device):
        return tuple([torch.clone(p).cuda(device) for p in layer_params])
    def clone_layers_params(layers_params, device):
        return [clone_layer_params(l, device) for l in layers_params]
    gpus_layers_params = [clone_layers_params(layers_params, i) for i in range(nGPUs)] 
    cpus_seeds = [t.reshape(-1) for t in seeds.reshape((-1, nGPUs)).chunk(nGPUs, dim=1)]

    processes = []
    for rank in range(nGPUs):
        p = mp.Process(target=init_process, args=(rank, gpus_layers_params[rank], cpus_seeds[rank], batch_size, train_process_ddp))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return gpus_layers_params[0]

# FSDP
def train_process_fsdp(local_rank, chunked_layers_params, seeds, batch_size):
    chunked_layer_params = chunked_layers_params[0]
    gen=torch.Generator()

    sharded_p0 = chunked_layer_params[0]
    sharded_p0s = [torch.zeros_like(sharded_p0).cuda(local_rank) for _ in range(nGPUs)]
    dist.all_gather(sharded_p0s, sharded_p0) 
    sharded_p1 = chunked_layer_params[1]
    sharded_p1s = [torch.zeros_like(sharded_p1).cuda(local_rank) for _ in range(nGPUs)]
    dist.all_gather(sharded_p1s, sharded_p1)
    
    layer_params = (torch.cat(sharded_p0s), torch.cat(sharded_p1s, dim=1))
    model_size = chunked_layer_params[0].shape[1]

    for seed in seeds.numpy().tolist():
        # get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)
        dloss_dx = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)

        # Forward
        y = tlayer_ffn_fwd(layer_params, x)

        # Backward
        dloss_dp = tlayer_ffn_bkwd(dloss_dx, layer_params, x)[1]
        chunked_dloss_dp = tuple([torch.clone(p).cuda(local_rank) for p in chunked_layer_params])
        def chunk_g(g, dim=0):
            return [ch_p.contiguous() for ch_p in g.chunk(nGPUs, dim=dim)]
        dist.reduce_scatter(chunked_dloss_dp[0], chunk_g(dloss_dp[0]), op=dist.ReduceOp.SUM)
        dist.reduce_scatter(chunked_dloss_dp[1], chunk_g(dloss_dp[1], dim=1), op=dist.ReduceOp.SUM)
        
        # Optimzer (in place)
        for param, grad in zip(chunked_layer_params, (chunked_dloss_dp[0], chunked_dloss_dp[1])):
            param.add_(-LR*grad)
  
def train_fsdp(layers_params, seeds, batch_size):
    assert len(seeds) % nGPUs == 0

    def chunk_p(p, dim):
        return [p_chunk.cuda(i) for i, p_chunk in enumerate(p.chunk(nGPUs, dim=dim))]
    def chunk_l(l):
        return [chunk_p(p, dim=i%2) for i, p in enumerate(l)]
    chunked_layers_params = [chunk_l(l) for l in layers_params]
    pre_gpus_layers_params = [list(map(list, zip(*chunked_l))) for chunked_l in chunked_layers_params]
    concat_gpu_layers = lambda i: [l[i] for l in pre_gpus_layers_params]
    gpus_layers_params = [concat_gpu_layers(i) for i in range(nGPUs)]
    cpus_seeds = [t.reshape(-1) for t in seeds.reshape((-1, nGPUs)).chunk(nGPUs, dim=1)]

    processes = []
    for rank in range(nGPUs):
        p = mp.Process(target=init_process, args=(rank, gpus_layers_params[rank], cpus_seeds[rank], batch_size, train_process_fsdp))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # TODO: support L>1
    gpus_layer_params = [gpus_layers_params[i][0] for i in range(nGPUs)]
    return [(torch.cat([gpu_p[0].cuda(0) for gpu_p in gpus_layer_params]), torch.cat([gpu_p[1].cuda(0) for gpu_p in gpus_layer_params], dim=1))]
    
#### Setup:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iters', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-d', '--model_size', type=int, default=4)
    parser.add_argument('-m', '--method', type=int, default=0)
    args = parser.parse_args()
    
    print(f'ARGS:\n iters:{args.iters}\n BS:{args.batch_size}\n D:{args.model_size}\n')

    seeds = torch.randint(100_000, (args.iters,)) # mock for dataset
    layers_params = [init_tlayer_ffn(args.model_size, 4*args.model_size) for _ in range(args.layers)]

    num_params = sum([p.numel() for layer_params in layers_params for p in layer_params ])
    def _gb(t_numel):
        return 4*t_numel/(1024 * 1024 * 1024)
    print(f'PARAMS: {num_params:_} (size {_gb(num_params)} GB)')
    print(f'\n')

    
    print(f'initial layers_params[0]', layers_params[0][0].shape, layers_params[0][1].shape)
    print(f'initial layers_params[0]', layers_params[0][0][:5,:5], layers_params[0][1][:5,:5])
    
    fns = [train_1gpu, train_ddp, train_fsdp]
    fns_layers_params = []
    mp.set_start_method('spawn')
    for i, fn in enumerate(fns):
        if args.method==0 or args.method==i+1:
            t0 = time.time()
            fn_layers_params = fn(layers_params, seeds, args.batch_size) # TODO: move to CPU
            t1 = time.time()
            fns_layers_params.append(fn_layers_params)
            print(f'\n{fn.__name__} takes {t1-t0} seconds')
            print(f'final {fn.__name__} layers_params[0]', fn_layers_params[0][0].shape, fn_layers_params[0][1].shape)
            print(f'final {fn.__name__} layers_params[0]', fn_layers_params[0][0][:5,:5], fn_layers_params[0][1][:5,:5])

    if args.method==0: # Compare DDP against FSDP(with DDP)
        assert torch.allclose(fns_layers_params[1][0][0], fns_layers_params[2][0][0]), f"ddp[0] {fns_layers_params[1][0][0]} fsdp[0] {fns_layers_params[2][0][0]}"
        assert torch.allclose(fns_layers_params[1][0][1], fns_layers_params[2][0][1]), f"ddp[1] {fns_layers_params[1][0][1]} fsdp[1] {fns_layers_params[2][0][1]}"
