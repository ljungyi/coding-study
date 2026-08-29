# Baseline:
# Encoder computed once.
# Decoder recomputes the entire generated prefix every decoding step.

from Transformer.transformer import Encoder,Decoder,Transformer
from Transformer.transformer import get_dec_enc_mask,get_dec_mask,get_enc_mask
import torch
import torch.nn as nn


src_vocab={
    "<PAD>": 0,
    "<UNK>": 1,
    "i": 2,
    "love": 3,
    "apples": 4,
    "eat": 5,
}

tgt_vocab = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "我": 4,
    "喜欢": 5,
    "苹果": 6,
    "吃": 7,
}
def encode_src(sentence):
    tokens=sentence.lower().split()

    return[src_vocab.get(token,1) for token in tokens]
           


def encode_tgt(sentence):
    tokens = sentence.split()

    return [
        tgt_vocab.get(
            token,
            tgt_vocab["<UNK>"]
        )
        for token in tokens
    ]

def pad_seq(seqs,pad_id):
    lengths=torch.tensor(
        [len(seq) for seq in seqs],
        dtype=torch.long
    )
    max_len=max(lengths).item()
    output=torch.full((len(seqs),max_len),pad_id,dtype=torch.long)

    for i in range(len(seqs)):
        output[i,:len(seqs[i])]=torch.tensor(seqs[i],dtype=torch.long)
    
    return lengths,output

pairs = [
    ("i love apples", "我 喜欢 苹果"),
    ("i eat", "我 吃"),
]
id_to_tgt={
        id:token
        for token,id in tgt_vocab.items()
    }
src_seq=[encode_src(src) for src,tgt in pairs]
tgt_seq=[encode_tgt(tgt) for src,tgt in pairs]

eos=tgt_vocab["<EOS>"]
bos=tgt_vocab["<BOS>"]

dec_seq=[[bos]+seq for seq in tgt_seq]
label_seq=[seq+[eos] for seq in tgt_seq]
dec_pad = tgt_vocab["<PAD>"]

dec_lens, dec_input_ids = pad_seq(
    dec_seq,
    dec_pad
)

tgt_lens, target_ids = pad_seq(
    label_seq,
    dec_pad
)


src_pad = src_vocab["<PAD>"]

src_lens, src_input_ids = pad_seq(
    src_seq,
    src_pad
)

torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available()
                    else "cpu")
print("device =", device)

d_model=16
d_ffn=d_model*4
num_heads=4
num_layers=4
max_len=16
dropout=0
max_new_tokens=10
batch_size=src_input_ids.size(0)

encoder=Encoder(max_len,num_layers,d_model,
                d_ffn,num_heads,
                dropout,dropout,dropout)
decoder=Decoder(d_model,num_heads,num_layers,
                max_len,d_ffn,dropout,
                dropout,dropout,len(tgt_vocab))
frontend=nn.Embedding(len(src_vocab),d_model)
model=Transformer(frontend,encoder,decoder,d_model,
                len(tgt_vocab))
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
        model.eval()
        enc_input=model.frontend(src_input_ids)
        enc_out=model.encoder(enc_input,enc_mask) 
        generated=torch.full((batch_size,1),bos,dtype=torch.long,device=device)
        gen_len=torch.ones(batch_size,dtype=torch.long,device=device)
        finished=torch.zeros(batch_size,dtype=torch.long,device=device).to(torch.bool)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                gen_max_len=gen_len.max()
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
                next_token=torch.where(active,next_token,torch.full_like(next_token,dec_pad))
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


    