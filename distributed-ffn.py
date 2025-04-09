### PARAMS + MODEL

import math
import torch

### PARAMS 

DEVICE = "cuda"

def init_linear_layer(m, n, scale=2e-2): 
    return scale * torch.randn((n, m), device=DEVICE)
    
def init_tlayer_ffn(emb_dim, ffn_dim):
    return [init_linear_layer(emb_dim, ffn_dim)] +  [init_linear_layer(ffn_dim, emb_dim)]


### MODEL

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

#### Setup:

BS, D = 4, 1 #4, 1 #8, 4 #32, 16
FFN = 2 * D #4 * D
x = torch.randn((BS, D), device="cuda")
dloss_dx = torch.randn((BS, D), device="cuda")
layer_params = init_tlayer_ffn(D, FFN)

#### BASELINE 1GPU

# Forward
y = tlayer_ffn_fwd(layer_params, x)
#print('y', y.shape, y[:10, :10])

# Backward
_, dloss_dp = tlayer_ffn_bkwd(dloss_dx, layer_params, x)
#print('dloss_dx', dloss_dx.shape, dloss_dx[:10, :10])
print(f'dloss_dp[0]', dloss_dp[0])
#print(f'dloss_dp[1]', dloss_dp[1])
#print('dloss_dp[0]', dloss_dp[0].shape, dloss_dp[0][:10, :10])
#print('dloss_dp[1]', dloss_dp[1].shape, dloss_dp[1][:10, :10])

# Optimizer step (just SGD for now)
n_layer_params_1gpu =tuple([p-0.001*g for p, g in zip(layer_params, dloss_dp)])
#print(f'n_layer_params_1gpu[0]', n_layer_params_1gpu[0].shape, n_layer_params_1gpu[0][:10, :10])
#print(f'n_layer_params_1gpu[1]', n_layer_params_1gpu[1].shape, n_layer_params_1gpu[1][:10, :10])

################### MULTI-GPU ########################
import torch.cuda.nccl as nccl
import multiprocessing as mp

nGPUs = torch.cuda.device_count()
if nGPUs==1:
    raise Exception("Only 1GPU available")
assert BS % nGPUs == 0

def clone_layer_params(layer_params, device):
    return tuple([torch.clone(p).cuda(device) for p in layer_params])
gpus_layer_params = [clone_layer_params(layer_params, i) for i in range(nGPUs)] 
gpus_x = torch.chunk(x, nGPUs, dim=0)
gpus_x = [gpu_x.cuda(i) for i, gpu_x in enumerate(gpus_x)]
gpus_dloss_dx = torch.chunk(dloss_dx, nGPUs, dim=0)
gpus_dloss_dx = [gpu_dloss_dx.cuda(i) for i, gpu_dloss_dx in enumerate(gpus_dloss_dx)]

gpus_dloss_dp = [tlayer_ffn_bkwd(gpu_dloss_dx, gpu_layer_params, gpu_x)[1] for gpu_dloss_dx, gpu_layer_params, gpu_x  in zip(gpus_dloss_dx, gpus_layer_params, gpus_x)]
gpus_ffn1_dloss_dp = [dloss_dp[0] for dloss_dp in gpus_dloss_dp]
print("BEFORE REDUCE gpus_ffn1_dloss_dp", gpus_ffn1_dloss_dp)
gpus_ffn2_dloss_dp = [dloss_dp[1] for dloss_dp in gpus_dloss_dp]
nccl.all_reduce(gpus_ffn1_dloss_dp)
print("AFTER REDUCE gpus_ffn1_dloss_dp", gpus_ffn1_dloss_dp)
nccl.all_reduce(gpus_ffn2_dloss_dp)
assert torch.allclose(gpus_ffn1_dloss_dp[0], dloss_dp[0]), f"gpus_ffn1_dloss_dp[0] {gpus_ffn1_dloss_dp[0]} dloss_dp[0] {dloss_dp[0]}"
assert torch.allclose(gpus_ffn2_dloss_dp[0], dloss_dp[1]), f"gpus_ffn1_dloss_dp[1] {gpus_ffn1_dloss_dp[1]} dloss_dp[1] {dloss_dp[1]}"

#### DDP
