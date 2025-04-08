import torch
import torch.cuda.nccl as nccl

nGPUs = torch.cuda.device_count()
dtype = torch.float
N=128

# AllGather
cpu_inputs = [torch.zeros(N).uniform_().to(dtype=dtype) for i in range(nGPUs)]
expected = torch.cat(cpu_inputs, 0)

inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
outputs = [
    torch.zeros(N * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
]
nccl.all_gather(inputs, outputs)

for tensor in outputs:
    assert torch.equal(tensor.cpu(), expected)


# AllReduce
expected = torch.sum(torch.stack(cpu_inputs), dim=0)
inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
nccl.all_reduce(inputs)
for tensor in inputs:
    assert torch.allclose(tensor.cpu(), expected), f"tensor.cpu() {tensor.cpu()}, expected {expected}"

# ReduceScatter
cpu_inputs = [torch.zeros(N*nGPUs).uniform_().to(dtype=dtype) for i in range(nGPUs)]
inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
outputs = [
    torch.zeros(N, device=i, dtype=dtype) for i in range(nGPUs)
]
nccl.reduce_scatter(inputs, outputs)
expected = torch.sum(torch.stack(cpu_inputs), dim=0)
output = torch.cat([output.cpu() for output in outputs], 0)
assert torch.allclose(output, expected), f"output.cpu() {output.cpu()}, expected {expected}"