import torch

x=torch.tensor([[1.,2.],
                [2.,3.]],
                dtype=torch.float16)
w=torch.tensor([[2.,3.,5.4],
                [6.7,8.0,9.8]],
                dtype=torch.float16)
y=x@w

loss=torch.sum(y**3)

dy=3*(y**2)
dw_theory=x.T@dy
print(f"dw00 theory num is {dw_theory[0][0]}")

eps=1e-4
w[0][0]+=eps
y=x@w
loss1=torch.sum(y**3)
w[0][0]-=2*eps
y=x@w
loss2=torch.sum(y**3)
dw_numercial=(loss1-loss2)/(2*eps)

print(f"dw00 numercial num is {dw_numercial}")