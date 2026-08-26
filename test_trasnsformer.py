from transformer import Encoder,Decoder,Transformer

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

    return([src_vocab.get(token,1)]
           for token in tokens)


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

src_seq=[encode_src(src) for src,tgt in pairs]
tgt_seq=[encode_tgt(tgt) for src,tgt in pairs]