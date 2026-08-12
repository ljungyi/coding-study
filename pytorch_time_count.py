import torch
from time import perf_counter

x=torch.randn(4096,4096)
y=torch.randn(4096,2048)

torch.cuda.synchronize();
start=perf_counter();

z=x@y

torch.cuda.synchronize()
time=perf_counter()-start

print(f"time is {time}s")
