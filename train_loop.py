import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_size = 2000
feature=30
category=4

torch.manual_seed(0)
x=torch.randn(sample_size,feature)
true_w=torch.randn(feature,category)
true_logits=x@true_w
y=true_logits.argmax(dim=1)

datset=TensorDataset(x,y)

dataloader=DataLoader(
    datset,
    batch_size=32,
    shuffle=True)

model=nn.Sequential(
    nn.Linear(feature,64),
    nn.ReLU(),
    nn.Linear(64,category)
)

model=model.to(device)

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
)

epochmax=6

for epoch in range(epochmax):

    model.train()
    run_loss=0

    for x,y in dataloader:
       x=x.to(device)
       y=y.to(device)
       optimizer.zero_grad()
       logits=model(x)
       loss=criterion(
           logits,
           y,
       )
       loss.backward()
       optimizer.step()
       print(f"loss = {loss}")

