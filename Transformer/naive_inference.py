# Baseline:
# Encoder computed once.
# Decoder recomputes the entire generated prefix every decoding step.

import time
from generate import generate
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
 _,_
)=build_data()
bos = tgt_vocab["<BOS>"]
eos = tgt_vocab["<EOS>"]
pad = tgt_vocab["<PAD>"]


model=init_model()
state_dict=torch.load("Transformer/checkpoint/toy_transformer.pt",map_location=device)
model.load_state_dict(state_dict)
model=model.to(device)

src_input_ids=src_input_ids.to(device)
max_new_tokens=1000
torch.cuda.synchronize()
start=time.time()
_=generate(model,device,src_input_ids,src_lens,
           max_new_tokens,
           bos,pad,eos,
           True)
torch.cuda.synchronize()
end=time.time()
print(f"consumed {end-start}s") 
                


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


    