import torch
N=10000

def sumf(dtype,reverse):
    values=([1.0]+[1e-3]*N)
    if(reverse):
        values.reverse()
    total=torch.tensor(
        0.0,
        dtype=dtype
    )
    for value in values:
        total=total+torch.tensor(value,dtype=dtype)
    return total.item()

for dtype in [torch.float16,
              torch.bfloat16,
              torch.float32,
              torch.float64]:
    sum1=sumf(dtype,False)
    print(f"datatype {dtype} reverse = False sum={sum1}")
    sum1=sumf(dtype,True)
    print(f"datatype {dtype} reverse = True sum={sum1}")

