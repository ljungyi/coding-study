# Baseline:
# Encoder computed once.
# Decoder recomputes the entire generated prefix every decoding step.

from transformer import get_dec_enc_mask,get_dec_mask,get_enc_mask
import torch
import torch.nn as nn
from data import build_data,tgt_vocab
from model_setup import init_model
import os

torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available()
                    else "cpu")
print("device =", device)


(src_input_ids,src_lens,
 dec_input_ids,dec_lens,
 target_ids,target_lens)=build_data()
batch_size=src_input_ids.size(0)

model=init_model()
model=model.to(device)
src_input_ids=src_input_ids.to(device)
dec_input_ids=dec_input_ids.to(device)
target_ids=target_ids.to(device)

criterion=nn.CrossEntropyLoss(
    ignore_index=tgt_vocab["<PAD>"]
)
optim=torch.optim.AdamW(model.parameters(),lr=1e-3,
                        weight_decay=0.0)
num_steps = 500
max_new_tokens=10
enc_max_len=src_input_ids.size(1)
dec_max_len=dec_input_ids.size(1)
enc_mask=get_enc_mask(batch_size,
                      enc_max_len,
                      src_lens,
                      device)
dec_mask=get_dec_mask(
    batch_size,dec_max_len,dec_lens,device
)
dec_enc_mask=get_dec_enc_mask(batch_size,enc_max_len,src_lens,
                              dec_max_len,device)

for step in range(num_steps):
    model.train()
    optim.zero_grad()
    enc_input=model.frontend(src_input_ids)
    enc_out=model.encoder(enc_input,enc_mask)   
    dec_hidden_states=model.decoder(dec_input_ids,enc_out,dec_mask,dec_enc_mask)
    logits=model.LM_head(dec_hidden_states)
    loss=criterion(
        logits.reshape(-1,len(tgt_vocab)),
        target_ids.reshape(-1)
        )
    loss.backward()
    optim.step()
    if step%100==0:
        print(f"loss = {loss.item()}")
        print(f"step = {step}")


os.makedirs("Transformer/checkpoint",exist_ok=True)
torch.save(model.state_dict(),"Transformer/checkpoint/toy_transformer.pt")
print(
    "checkpoint saved:"
    " Transformer/checkpoints/toy_transformer.pt"
)


       