import torch
import multiprocessing as mp

nGPUs = torch.cuda.device_count()


def init_pool_processes(the_barrier, the_gpus_outputs):
    global barrier
    barrier = the_barrier
    global gpus_outputs
    gpus_outputs = the_gpus_outputs

def add_rank(local_rank):
    #gpus_outputs[local_rank].add_(local_rank)
    gpus_outputs[local_rank][0].add_(local_rank)
    gpus_outputs[local_rank][1].add_(local_rank)
    barrier.wait()
    if local_rank==0:
        print("Inside", gpus_outputs)
    
if __name__ == '__main__':
    
    mp.set_start_method('spawn')

    #output = torch.zeros((2,2))
    #gpus_output= [torch.clone(output).cuda(i) for i in range(nGPUs)] 
    output = (torch.zeros((2,2)), torch.zeros((2,2)))
    def clone_output(output, i):
        return (torch.clone(output[0]).cuda(i), torch.clone(output[1]).cuda(i))
    gpus_output= [clone_output(output, i) for i in range(nGPUs)] 
    
    barrier = mp.Barrier(nGPUs)
    with mp.Pool(nGPUs, initializer=init_pool_processes, initargs=(barrier, gpus_output)) as p:
        p.map(add_rank, range(nGPUs))
    print("Outside", gpus_output)