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

BS, D = 32, 16
FFN = 4 * D
x = torch.randn((BS, D), device="cuda")
dloss_dx = torch.randn((BS, D), device="cuda")
layer_params = init_tlayer_ffn(D, FFN)

#### BASELINE 1GPU

# Forward
y = tlayer_ffn_fwd(layer_params, x)
print('y', y.shape, y[:10, :10])

# Backward
dloss_dx, dloss_dp = tlayer_ffn_bkwd(dloss_dx, layer_params, x)
print('dloss_dx', dloss_dx.shape, dloss_dx[:10, :10])
print('dloss_dp[0]', dloss_dp[0].shape, dloss_dp[0][:10, :10])
print('dloss_dp[1]', dloss_dp[1].shape, dloss_dp[1][:10, :10])

# Optimizer step (just SGD for now)
n_layer_params =tuple([p-0.001*g for p, g in zip(layer_params, dloss_dp)])
print(f'n_layer_params[0]', n_layer_params[0].shape, n_layer_params[0][:10, :10])
print(f'n_layer_params[1]', n_layer_params[1].shape, n_layer_params[1][:10, :10])