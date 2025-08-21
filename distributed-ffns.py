# This is a toy example how to distribute the computation of Transformer's FFN sublocks among GPUs
# It shows how to implmement DDP and FSDP from the first principle (see the asterisk below).
#
# To test, run "python distributed-ffns.py --iters 16 --batch_size 8192 --layers 1 --model_size 8192 --method M", where M is one of:
#   "0": run all methods;  "1": run on 1GPU, "2": run DDP, "3": run FSDP (with DDP)
#
# To see the advantage of FSDP over DDP, one can check the following config if running on 4 GPUs with 24GB memory each:
# "python distributed-ffns.py --iters 4 --batch_size 6144 --model_size 8192 --layers 8 --method 3".
# This will result in over 4B model (16GB of space, as fp32 is used). The training will work if FSDP is used (i.e. method 3), but not with DDP (i.e. method 2).
#
# NB: For simplicity, the random dataset is used, and no real loss function is used ( I imitate it by randomized dloss_dx coming from "right")
#
# Remaining TODO: improve the overalapping of communication and computation for FSDP (ReduceScatter is not overlapped right now. We need more than one process group, see: https://github.com/pytorch/pytorch/issues/67158)
#
# (The asterisk: I use the NCCL through torch.distributed package i.e. I use its init_process_group() method and its communication collectives e.g. all_reduce.
# I meant to use torch.cuda.nccl directly, but there is a known issue: https://github.com/pytorch/pytorch/issues/38019.
# Using torch.distributed package slightly simplifies coding up the communication between GPUs, but this is still relatively a thin wrapper for NCCL.)

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
DLOSS_DX_COEF = 0.1 # We imitatate loss function by randomized dloss_dx. We may want to scale them depending on the model size&iterations


### PARAMS + MODEL

def init_linear_layer(m, n, scale=2e-2): # no bias as simplification
    return scale * torch.randn((n, m))
    
def init_tlayer_ffn(emb_dim, ffn_dim):
    return [init_linear_layer(emb_dim, ffn_dim)] +  [init_linear_layer(ffn_dim, emb_dim)]

def linear_fwd(layer_params, x): # input: seq_len x emb_dim
    return torch.matmul(x, torch.transpose(layer_params, 0, 1))

def t_linear_bkwd(dloss_dx, layer_params, x): # input: N x D
    return torch.einsum('bc, bd -> cd', dloss_dx, x), torch.einsum('bc, cd -> bd', dloss_dx, layer_params)

def t_relu_fwd(x):
    return torch.where(torch.le(x, 0), 0, x) # as inputs are broadcastable in where&le - follows pytorch's implementation

def t_relu_bkwd_(dloss_dx, x): # NB: in place bkwd
    dloss_dx.masked_fill_(x <= 0, 0)
    return dloss_dx

def tlayer_ffn_fwd(layer_params, x): # input: seq_len x emb_dim
    x = linear_fwd(layer_params[0], x)
    x = t_relu_fwd(x)
    x = linear_fwd(layer_params[1], x)
    return x


def tlayer_ffn_bkwd(dloss_dx, layer_params, x):
    x_in = x
    x_before_act = linear_fwd(layer_params[0], x)
    x = t_relu_fwd(x_before_act)
    
    # propagate back
    ffn2_dloss_dp, dloss_dx = t_linear_bkwd(dloss_dx, layer_params[1], x)
    dloss_dx = t_relu_bkwd_(dloss_dx, x_before_act)
    ffn1_dloss_dp, dloss_dx = t_linear_bkwd(dloss_dx, layer_params[0], x_in)

    return dloss_dx.reshape(x_in.shape), (ffn1_dloss_dp, ffn2_dloss_dp)

#### Training methods: 1GPU, DDP, FSDP

## 1 GPU

def train_1gpu(layers_params, seeds, batch_size):
    gen=torch.Generator()
    model_size = layers_params[0][0].shape[1]

    move_l = lambda l: [p.cuda(0) for p in l]
    layers_params = [move_l(l) for l in layers_params]

    for seed in seeds.numpy().tolist():
        # Get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(0)
        dloss_dx = DLOSS_DX_COEF*torch.randn((batch_size, model_size), generator=gen).cuda(0) 
        
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

## Multi-GPU

# Utils
def init_process(rank, layers_params, seeds, batch_size, fn):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=nGPUs)
    
    fn(rank, layers_params, seeds, batch_size)
    
from torch.profiler import profile, ProfilerActivity
def torch_profile_rank_0(func):
    global wrapper_torch_profile_rank_0 # TODO/Q: Probably not safe if we use this wrapper more than once?
    def wrapper_torch_profile_rank_0(*args, **kwargs):
        local_rank =args[0] # assume first parameter is local_rank
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True,
                     with_stack=True) as prof:
            func(*args, **kwargs)
        if local_rank==0:
            prof.export_chrome_trace("trace_profiler_trace.json")
            print("Profiler exported")
    return wrapper_torch_profile_rank_0
    
# DDP
#@torch_profile_rank_0
def train_process_ddp(local_rank, layers_params, seeds, batch_size):
    gen=torch.Generator()
    model_size = layers_params[0][0].shape[1]

    for seed in seeds.numpy().tolist():
        # get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)
        dloss_dx = DLOSS_DX_COEF*torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)

        # Forward
        y=x
        acts = []
        for l in layers_params:
            acts.append(y)
            y = tlayer_ffn_fwd(l, y)

        # Backward + optimizer (in-place SGD)
        batch_dloss_dx = dloss_dx
        dloss_dp = None
        handles = []
        def update_l_params(i):
            for h in handles:
                h.wait()
            for param, grad in zip(layers_params[i], (dloss_dp[0], dloss_dp[1])):
                param.add_(-LR*grad)
                
        for i in reversed(range(len(layers_params))):
            batch_dloss_dx, n_dloss_dp = tlayer_ffn_bkwd(batch_dloss_dx, layers_params[i], acts[i])  
            if i< len(layers_params)-1:
                update_l_params(i+1)
            dloss_dp = n_dloss_dp
            handles = [dist.all_reduce(dloss_dp[j], op=dist.ReduceOp.SUM, async_op=True) for j in range(len(dloss_dp))]
        update_l_params(0)
  
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
#@torch_profile_rank_0
def train_process_fsdp(local_rank, chunked_layers_params, seeds, batch_size):
    gen=torch.Generator()
    layers = len(chunked_layers_params)
    model_size = chunked_layers_params[0][0].shape[1]
    
    def gather_layer_params_start(l):
        sharded_p0 = chunked_layers_params[l][0]
        sharded_p0s = [torch.zeros_like(sharded_p0).cuda(local_rank) for _ in range(nGPUs)]
        handle0 = dist.all_gather(sharded_p0s, sharded_p0, async_op=True) 
        sharded_p1 = chunked_layers_params[l][1]
        sharded_p1s = [torch.zeros_like(sharded_p1).cuda(local_rank) for _ in range(nGPUs)]
        handle1 = dist.all_gather(sharded_p1s, sharded_p1, async_op=True)

        return ([sharded_p0s, sharded_p1s], [handle0, handle1])
        
    def gather_layer_params_end(sharded_ps, handles):
        for h in handles:
            h.wait()
        return (torch.cat(sharded_ps[0]), torch.cat(sharded_ps[1], dim=1))
        
    chunked_dloss_dp = tuple([torch.clone(p).cuda(local_rank) for p in chunked_layers_params[0]]) # Buffers for results of ReduceScatter
    

    for seed in seeds.numpy().tolist():
        # get data
        gen.manual_seed(seed)
        x = torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)
        dloss_dx = DLOSS_DX_COEF*torch.randn((batch_size, model_size), generator=gen).cuda(local_rank)

        # Forward
        y=x
        acts = []
        sharded_ps, handles = gather_layer_params_start(0)
        layer_params = gather_layer_params_end(sharded_ps, handles)
                
        for l in range(layers):
            acts.append(y)
            if l< layers-1:
                sharded_ps, handles = gather_layer_params_start(l+1)
            y = tlayer_ffn_fwd(layer_params, y)
            if l< layers-1:
                layer_params = gather_layer_params_end(sharded_ps, handles)

        # Backward + optimizer (in-place SGD)
        batch_dloss_dx = dloss_dx
        for i in reversed(range(layers)):
            if i>0:
                sharded_ps, handles = gather_layer_params_start(i-1)
            batch_dloss_dx, dloss_dp = tlayer_ffn_bkwd(batch_dloss_dx, layer_params, acts[i])  
            if i>0:
                layer_params = gather_layer_params_end(sharded_ps, handles)

            # TODO: overlap communication and computation for the below (we need more than one ProcessGroup)
            def chunk_g(g, dim=0):
                return [ch_p.contiguous() for ch_p in g.chunk(nGPUs, dim=dim)]
            dist.reduce_scatter(chunked_dloss_dp[0], chunk_g(dloss_dp[0]), op=dist.ReduceOp.SUM)
            dist.reduce_scatter(chunked_dloss_dp[1], chunk_g(dloss_dp[1], dim=1), op=dist.ReduceOp.SUM)

            for param, grad in zip(chunked_layers_params[i], (chunked_dloss_dp[0], chunked_dloss_dp[1])):
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

    def concat_l(l):
        gpus_layer_params = [gpus_layers_params[i][l] for i in range(nGPUs)]
        return (torch.cat([gpu_p[0].cuda(0) for gpu_p in gpus_layer_params]), torch.cat([gpu_p[1].cuda(0) for gpu_p in gpus_layer_params], dim=1))
    return [concat_l(l) for l in range(len(layers_params))]
    
#### Setup:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iters', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-d', '--model_size', type=int, default=4)
    parser.add_argument('-m', '--method', type=int, default=0)
    args = parser.parse_args()
    
    print(f'ARGS:\n iters: {args.iters}\n BS: {args.batch_size}\n D: {args.model_size}\n FFN: 4*D\n')

    seeds = torch.randint(100_000, (args.iters,)) # seeds for the random (mocked) dataset
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
        for l in range(args.layers):
            if not torch.allclose(fns_layers_params[1][l][0], fns_layers_params[2][l][0]):
                print(f"SoftAssertionError: L {l} ddp[0] {fns_layers_params[1][l][0]} fsdp[0] {fns_layers_params[2][l][0]}")
            if not torch.allclose(fns_layers_params[1][l][1], fns_layers_params[2][l][1]): 
                print(f"SoftAssertionError: L {l} ddp[1] {fns_layers_params[1][l][1]} fsdp[1] {fns_layers_params[2][l][1]}")
