# Baseline:
# Encoder computed once.
# Decoder recomputes the entire generated prefix every decoding step.

from transformer import get_dec_enc_mask,get_dec_mask,get_enc_mask
import torch
import torch.nn as nn
from data import build_data,tgt_vocab
from model_setup import init_model

torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available()
                    else "cpu")
print("device =", device)


(src_input_ids,src_lens,
 dec_input_ids,dec_lens,
 target_ids,target_lens,
)=build_data()
bos = tgt_vocab["<BOS>"]
eos = tgt_vocab["<EOS>"]
pad = tgt_vocab["<PAD>"]
batch_size=src_input_ids.size(0)

model=init_model()
state_dict=torch.load("Transformer/checkpoint/toy_transformer.pt",map_location=device)
model.load_state_dict(state_dict)
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


model.eval()
enc_input=model.frontend(src_input_ids)
enc_out=model.encoder(enc_input,enc_mask) 
generated=torch.full((batch_size,1),bos,dtype=torch.long,device=device)
gen_len=torch.ones(batch_size,dtype=torch.long,device=device)
finished=torch.zeros(batch_size,dtype=torch.bool,device=device)
with torch.no_grad():
    for _ in range(max_new_tokens):
        gen_max_len=generated.size(1)
        gen_mask=get_dec_mask(
            batch_size,gen_max_len,gen_len,device
        )
        gen_enc_mask=get_dec_enc_mask(batch_size,enc_max_len,src_lens,
                        gen_max_len,device)
        gen_hidden_states=model.decoder(generated,enc_out,gen_mask,gen_enc_mask)
        logits=model.LM_head(gen_hidden_states)
        # logits=model.forward(src_input_ids,
        #                     src_lens,
        #                     gen_len,
        #                     generated)
        next_logits=logits[:,-1,:]
        ##重要！
        next_token=next_logits.argmax(dim=-1)
        active=~finished
        next_token=torch.where(active,next_token,torch.full_like(next_token,pad))
        generated=torch.cat([generated,next_token.unsqueeze(1)],dim=1)
        gen_len=gen_len+active.long()
        finished=finished | (next_token==eos)
        if finished.all():
            break
print("generated ids:")
print(generated)   
                


        # model.eval()
        # with torch.no_grad():
        #     logits=model.forward(src_input_ids,src_lens,
        #                         tgt_lens,dec_input_ids)
        #     predict_ids=logits.argmax(dim=-1)
        #     valid_ids=(
        #         target_ids!=dec_pad
        #     )
        #     accuracy=(
        #         predict_ids[valid_ids]
        #         ==target_ids[valid_ids]
        #     ).float().mean()
        # print(f"loss = {loss.item()}")
        # print(f"step = {step}")
        # print(f"accuracy = {accuracy}")
    
        # for i in range(predict_ids.size(0)):
        #     print("target:",
        #         [id_to_tgt[x.item()] for x in target_ids[i]])

        #     print("pred:  ",
        #         [id_to_tgt[x.item()] for x in predict_ids[i]])

#     print(
#     "dec_input_ids:",
#     dec_input_ids.shape
#     )

#     print(
#     "logits:",
#     logits.shape
#     )


    