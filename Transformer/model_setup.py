from transformer import Transformer,Encoder,Decoder
import torch.nn as nn
from data import tgt_vocab,src_vocab

d_model=16
d_ffn=d_model*4
num_heads=4
num_layers=4
max_len=16
dropout=0



def init_model():
    encoder=Encoder(max_len,num_layers,d_model,
                    d_ffn,num_heads,
                    dropout,dropout,dropout)
    decoder=Decoder(d_model,num_heads,num_layers,
                    max_len,d_ffn,dropout,
                    dropout,dropout,len(tgt_vocab))
    frontend=nn.Embedding(len(src_vocab),d_model)
    model=Transformer(frontend,encoder,decoder,d_model,
                    len(tgt_vocab))
    return model