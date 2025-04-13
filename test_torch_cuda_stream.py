import torch
import torch.cuda.nccl as nccl
nGPUs = torch.cuda.device_count()

M, N, K = 128, 64, 256
streams = [torch.cuda.Stream() for _ in range(nGPUs)]
a_tensors = [torch.randn((M, N), device=i) for i in range(nGPUs)]
b_tensors = [torch.randn((N, K), device=i) for i in range(nGPUs)]
outputs = [None] * nGPUs

# do I need torch.cuda.synchronize() here?

# import os
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29500'
# breakpoint()
# torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
#for i in range(nGPUs):
#    with torch.cuda.device(i):
#        torch.distributed.init_process_group(backend="nccl", world_size=nGPUs, rank=i)
        
# PyTorch's bug: https://github.com/pytorch/pytorch/issues/38019
# uid = nccl.unique_id()
# print(uid)
# comms = []
# for i in range(nGPUs):
#     with torch.cuda.device(i):
#         c = nccl.init_rank(nGPUs, uid, i)
#         comms.append(c)

for i, s_and_ts in enumerate(zip(streams, a_tensors, b_tensors)):
    s, a, b = s_and_ts
    with torch.cuda.stream(s):
        for j in range(10):
            outputs[i] = torch.matmul(a, b)
            # TODO make the below works
            #nccl.all_reduce(outputs, streams=streams)
torch.cuda.synchronize() # do I need this?
print(outputs)
