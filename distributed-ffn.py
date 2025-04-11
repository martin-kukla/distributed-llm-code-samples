import argparse
import math
import torch
import torch.cuda.nccl as nccl
import multiprocessing as mp
import time

nGPUs = torch.cuda.device_count()
if nGPUs==1:
    raise Exception("Only 1GPU available")

LR = 0.0001 # 0.1 for testing


### PARAMS + MODEL

def init_linear_layer(m, n, scale=2e-2): 
    return scale * torch.randn((n, m), device=0) # TODO: define on CPU instead - cuda complains
    
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

#### Training methods: 1GPU, DDP

def train_1gpu(dloss_dx, layer_params, x, steps=2):
    x = x.cuda(0)
    dloss_dx = dloss_dx.cuda(0)
    layer_params = [p.cuda(0) for p in layer_params]

    for _ in range(steps):
        # Forward
        y = tlayer_ffn_fwd(layer_params, x)
        
        # Backward
        _, dloss_dp = tlayer_ffn_bkwd(dloss_dx, layer_params, x)
        
        # Optimizer step (just SGD for now)
        layer_params = [p-LR*g for p, g in zip(layer_params, dloss_dp)]
    
    return layer_params


def tlayer_ffn_bkwd_wrapper(gpu_args):
        _, gpu_args, _ = gpu_args
        return tlayer_ffn_bkwd(gpu_args[0], gpu_args[1], gpu_args[2])[1]

def _train_ddp(dloss_dx, layer_params, x, steps=2):
    assert BS % nGPUs == 0

    mp.set_start_method('spawn')
    for _ in range(steps):
        def clone_layer_params(layer_params, device):
            return tuple([torch.clone(p).cuda(device) for p in layer_params])
        gpus_layer_params = [clone_layer_params(layer_params, i) for i in range(nGPUs)] 
        gpus_x = torch.chunk(x, nGPUs, dim=0)
        gpus_x = [gpu_x.cuda(i) for i, gpu_x in enumerate(gpus_x)]
        gpus_dloss_dx = torch.chunk(dloss_dx, nGPUs, dim=0)
        gpus_dloss_dx = [gpu_dloss_dx.cuda(i) for i, gpu_dloss_dx in enumerate(gpus_dloss_dx)]
        gpus_dloss_dp = [None]*nGPUs
    
        with mp.Pool(nGPUs) as p:
            gpus_args = zip(gpus_dloss_dx, gpus_layer_params, gpus_x)
            gpus_args = [(i, gpu_args, gpus_dloss_dp) for i, gpu_args in enumerate(gpus_args)]
            gpus_dloss_dp = p.map(tlayer_ffn_bkwd_wrapper, gpus_args)
        #gpus_dloss_dp = [tlayer_ffn_bkwd(gpu_dloss_dx, gpu_layer_params, gpu_x)[1] for gpu_dloss_dx, gpu_layer_params, gpu_x  in zip(gpus_dloss_dx, gpus_layer_params, gpus_x)]
    
        gpus_ffn1_dloss_dp = [dloss_dp[0] for dloss_dp in gpus_dloss_dp]
        gpus_ffn2_dloss_dp = [dloss_dp[1] for dloss_dp in gpus_dloss_dp]
        nccl.all_reduce(gpus_ffn1_dloss_dp)
        nccl.all_reduce(gpus_ffn2_dloss_dp)   
    
        # Optimizer step (just SGD for now)
        layer_params = [p-LR*g for p, g in zip(gpus_layer_params[0], (gpus_ffn1_dloss_dp[0], gpus_ffn2_dloss_dp[0]))]

    return layer_params

def train_ddp_process(gpu_args):
    local_rank, gpu_args = gpu_args
    dloss_dx, layer_params, x = gpu_args

    for _ in range(steps):
        dloss_dp = tlayer_ffn_bkwd(dloss_dx, layer_params, x)[1]
        # NB, for some reason, if I use set_, the results will not be visible to the other processes?!
        gpus_dloss_dp[local_rank][0].add_(dloss_dp[0])
        gpus_dloss_dp[local_rank][1].add_(dloss_dp[1])
        
        barrier.wait()
        if local_rank==0:
            gpus_ffn1_dloss_dp = [dloss_dp[0] for dloss_dp in gpus_dloss_dp]
            gpus_ffn2_dloss_dp = [dloss_dp[1] for dloss_dp in gpus_dloss_dp]
            nccl.all_reduce(gpus_ffn1_dloss_dp)
            nccl.all_reduce(gpus_ffn2_dloss_dp)   
        barrier.wait()
        
        layer_params = [p-LR*g for p, g in zip(layer_params, (gpus_dloss_dp[local_rank][0], gpus_dloss_dp[local_rank][1]))]
        gpus_dloss_dp[local_rank][0].zero_()
        gpus_dloss_dp[local_rank][1].zero_()

    return layer_params

def init_pool_processes(the_barrier, the_gpus_dloss_dp, the_steps):
    global barrier
    barrier = the_barrier
    global gpus_dloss_dp
    gpus_dloss_dp = the_gpus_dloss_dp
    global steps
    steps = the_steps
    
def train_ddp(dloss_dx, layer_params, x, steps=2):
    assert dloss_dx.shape[0] % nGPUs == 0

    def clone_layer_params(layer_params, device):
        return tuple([torch.clone(p).cuda(device) for p in layer_params])
    gpus_layer_params = [clone_layer_params(layer_params, i) for i in range(nGPUs)] 
    gpus_x = torch.chunk(x, nGPUs, dim=0)
    gpus_x = [gpu_x.cuda(i) for i, gpu_x in enumerate(gpus_x)]
    gpus_dloss_dx = torch.chunk(dloss_dx, nGPUs, dim=0)
    gpus_dloss_dx = [gpu_dloss_dx.cuda(i) for i, gpu_dloss_dx in enumerate(gpus_dloss_dx)]
    dloss_dp = tuple([torch.zeros_like(p) for p in layer_params])
    gpus_dloss_dp = [clone_layer_params(dloss_dp, i) for i in range(nGPUs)]

    mp.set_start_method('spawn')
    barrier = mp.Barrier(nGPUs)
    with mp.Pool(nGPUs, initializer=init_pool_processes, initargs=(barrier, gpus_dloss_dp, steps)) as p:
        gpus_args = zip(gpus_dloss_dx, gpus_layer_params, gpus_x)
        gpus_args = [(i, gpu_args) for i, gpu_args in enumerate(gpus_args)]
        gpus_layer_params = p.map(train_ddp_process, gpus_args)

    return gpus_layer_params[0]

#### Setup:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iters', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-d', '--model_size', type=int, default=4)
    parser.add_argument('-m', '--mode', type=int, default=0)
    args = parser.parse_args()

    x = torch.randn((args.batch_size, args.model_size))
    dloss_dx = torch.randn((args.batch_size, args.model_size))
    layer_params = init_tlayer_ffn(args.model_size, 4*args.model_size)
    

    if args.mode==0 or args.mode==1:
        t0 = time.time()
        n_layer_params_1gpu = train_1gpu(dloss_dx, layer_params, x, args.iters)
        t1 = time.time()
        print(f'n_layer_params_1gpu takes {t1-t0} seconds: ', n_layer_params_1gpu)

    if args.mode==0 or args.mode==2:
        t0 = time.time()
        n_layer_params_ddp = train_ddp(dloss_dx, layer_params, x, args.iters)
        t1 = time.time()
        print(f'n_layer_params_ddp takes {t1-t0} seconds: ', n_layer_params_ddp)

    if args.mode==0:
        assert torch.allclose(n_layer_params_ddp[0], n_layer_params_1gpu[0]), f"n_layer_params_ddp[0] {n_layer_params_ddp[0]} n_layer_params_1gpu[0] {n_layer_params_1gpu[0]}"
        assert torch.allclose(n_layer_params_ddp[1], n_layer_params_1gpu[1]), f"n_layer_params_ddp[1] {n_layer_params_ddp[1]} n_layer_params_1gpu[1] {n_layer_params_1gpu[1]}"
