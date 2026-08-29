import numpy as np
import torch
import torch.nn as nn


def get_enc_mask(
    batch: int,
    max_len: int,
    enc_lens: torch.Tensor,
    device: torch.device
) -> torch.Tensor:

    attn_mask = torch.ones(
        (batch, max_len, max_len),
        device=device
    )

    for i in range(batch):
        attn_mask[i, :, :enc_lens[i]] = 0

    return attn_mask.to(torch.bool)


def get_dec_mask(
    b: int,
    max_len: int,
    dec_lens:torch.Tensor,
    device: torch.device
) -> torch.Tensor:

    attn_mask = torch.ones(
        (b, max_len, max_len),
        device=device,
        dtype=torch.bool
    )

    attn_mask = torch.triu(
        attn_mask,
        diagonal=1
    )
    padding_mask=torch.ones(
        (b,max_len,max_len),
        device=device,
        dtype=torch.bool
    )
    for i in range(b):
        padding_mask[i,:,:dec_lens[i]]=False
    padding_mask=padding_mask | attn_mask

    return padding_mask


def get_dec_enc_mask(
    b: int,
    max_enc_len: int,
    enc_len: torch.Tensor,
    max_dec_len: int,
    device: torch.device
):

    attn_mask = torch.ones(
        (b, max_dec_len, max_enc_len),
        device=device,
    )

    for i in range(b):
        attn_mask[i, :, :enc_len[i]] = 0

    return attn_mask.to(torch.bool)


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_head: int,
        num_heads: int,
        d_model: int,
        p: float = 0.
    ):
        """
        d_model:
            每个 token 的完整 hidden dimension

        d_head:
            每个 attention head 的特征维度

        num_heads:
            attention head 数量

        通常：
            d_model = num_heads * d_head
        """

        super(MultiHeadAttention, self).__init__()

        self.d_head = d_head
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(p)

        # 线性投影
        self.W_Q = nn.Linear(
            d_model,
            d_head * num_heads
        )

        self.W_K = nn.Linear(
            d_model,
            d_head * num_heads
        )

        self.W_V = nn.Linear(
            d_model,
            d_head * num_heads
        )

        self.W_out = nn.Linear(
            d_head * num_heads,
            d_model
        )

        # 初始化投影矩阵
        nn.init.normal_(
            self.W_Q.weight,
            mean=0,
            std=np.sqrt(
                2.0 / (d_model + d_head)
            )
        )

        nn.init.normal_(
            self.W_K.weight,
            mean=0,
            std=np.sqrt(
                2.0 / (d_model + d_head)
            )
        )

        nn.init.normal_(
            self.W_V.weight,
            mean=0,
            std=np.sqrt(
                2.0 / (d_model + d_head)
            )
        )

        nn.init.normal_(
            self.W_out.weight,
            mean=0,
            std=np.sqrt(
                2.0 / (d_model + d_head)
            )
        )

    def forward(
        self,
        Xq: torch.Tensor,
        Xk: torch.Tensor,
        Xv: torch.Tensor,
        attn_mask: torch.Tensor
    ):

        """
        Xq:
            (batch, q_len, d_model)

        Xk, Xv:
            (batch, k_len, d_model)

        拆头之后：

        Q:
            (batch, num_heads, q_len, d_head)

        K,V:
            (batch, num_heads, k_len, d_head)
        """

        N = Xq.size(0)

        q_len = Xq.size(1)
        k_len = Xk.size(1)

        d_head = self.d_head
        num_heads = self.num_heads

        Q = (
            self.W_Q(Xq)
            .view(N, -1, num_heads, d_head)
            .transpose(1, 2)
        )

        K = (
            self.W_K(Xk)
            .view(N, -1, num_heads, d_head)
            .transpose(1, 2)
        )

        V = (
            self.W_V(Xv)
            .view(N, -1, num_heads, d_head)
            .transpose(1, 2)
        )

        logits = (
            torch.matmul(
                Q,
                K.transpose(-1, -2)
            )
            / np.sqrt(d_head)
        )

        if attn_mask is not None:

            assert attn_mask.size() == (
                N,
                q_len,
                k_len
            )

            attn_mask = (
                attn_mask
                .unsqueeze(1)
                .repeat(
                    1,
                    num_heads,
                    1,
                    1
                )
                .bool()
            )

            logits.masked_fill_(
                attn_mask,
                -1e4
            )

        attn = torch.softmax(
            logits,
            dim=-1
        )

        attn = self.dropout(attn)

        V = (
            torch.matmul(attn, V)
            .transpose(1, 2)
            .contiguous()
            .view(
                N,
                -1,
                num_heads * d_head
            )
        )

        V = self.W_out(V)
        V = self.dropout(V)


        return V


def Position_embedding(
    seq_len: int,
    d_model: int
):

    pos_mat = torch.zeros(
        seq_len,
        d_model
    )

    positions = torch.arange(
        0,
        seq_len
    )

    for i in range(d_model):

        f = (
            torch.sin
            if i % 2 == 0
            else torch.cos
        )

        pos_mat[:, i] = f(
            positions
            /
            np.pow(
                10000,
                2 * (i // 2) / d_model
            )
        )

    return pos_mat.float()


class FFN(nn.Module):

    def __init__(
        self,
        d_ffn: int,
        d_model: int,
        dropout=0.1
    ):

        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn

        self.W1 = nn.Linear(
            d_model,
            d_ffn,
        )

        self.W2 = nn.Linear(
            d_ffn,
            d_model
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        X: torch.Tensor
    ):

        assert X.size(-1) == self.d_model

        X = self.W1(X)
        X = self.relu(X)
        X = self.W2(X)
        X = self.dropout(X)

        return X


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        dropout_attn: float,
        dropout_ffn: float,
    ):

        super().__init__()

        assert d_model % num_heads == 0

        d_head = d_model // num_heads

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.encoder_attn = MultiHeadAttention(
            d_head,
            num_heads,
            d_model,
            dropout_attn
        )

        self.ffn = FFN(
            d_ffn,
            d_model,
            dropout_ffn
        )

    def forward(
        self,
        X: torch.Tensor,
        attn_mask: torch.Tensor
    ) -> torch.Tensor:

        residual = X

        context = self.encoder_attn(
            X,
            X,
            X,
            attn_mask
        )

        out1 = self.norm1(
            context + residual
        )

        residual = out1

        ffn_output = self.ffn(
            out1
        )

        output = self.norm2(
            ffn_output + residual
        )

        return output


class Encoder(nn.Module):

    def __init__(
        self,
        max_len: int,
        num_layers: int,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        dropout_attn: float,
        dropout_ffn: float,
        dropout_emb:float
    ):

        super().__init__()

        self.max_seq_len = max_len

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_ffn,
                    num_heads,
                    dropout_attn,
                    dropout_ffn,
                )
                for _ in range(num_layers)
            ]
        )

        self.pos_embed = (
            nn.Embedding.from_pretrained(
                Position_embedding(
                    self.max_seq_len,
                    d_model
                )
            )
        )

        self.dropout_emb = nn.Dropout(
            dropout_emb
        )


    def forward(
        self,
        X,
        enc_mask=None
    ):

        seq_len = X.size(1)

        out = (
            X
            +
            self.pos_embed(
                torch.arange(0, seq_len,device=X.device)
            )
        )

        out = self.dropout_emb(out)

        for layer in self.layers:
            out = layer(
                out,
                enc_mask
            )

        return out


class DecoderLayer(nn.Module):

    def __init__(
        self,
        max_len:int,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        dropout_attn: float,
        dropout_ffn: float,
    ):

        super().__init__()

        assert d_model % num_heads == 0

        d_head = d_model // num_heads
        self.max_len=max_len

        self.decoder_attn = (
            MultiHeadAttention(
                d_head,
                num_heads,
                d_model,
                dropout_attn
            )
        )
        self.cross_attn=MultiHeadAttention(d_head,
                                           num_heads,
                                           d_model,
                                           dropout_attn)

        self.ffn = FFN(
            d_ffn,
            d_model,
            dropout_ffn
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        

    def forward(self,X,enc_X,decoder_mask,dec_enc_mask):
        residual=X
        out=self.decoder_attn(X,X,X,decoder_mask)
        out=self.norm1(out+residual)

        residual=out
        out=self.cross_attn(out,enc_X,enc_X,dec_enc_mask)
        out=self.norm2(out+residual)

        residual=out
        out=self.ffn(out)
        out=self.norm3(out+residual)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 num_layers:int,
                 max_seq_len:int,
                 d_ffn:int,
                 dropout_attn:float,
                 dropout_ffn:float,
                 dropout_emb:float,
                 vocab_size:int
                 ):
        super().__init__()
        self.layers=nn.ModuleList([DecoderLayer(max_seq_len,
                                                d_model,
                                                d_ffn,
                                                num_heads,
                                                dropout_attn,
                                                dropout_ffn,
                                                )
                                                for _ in range(num_layers)])
        self.tgt_emb=nn.Embedding(vocab_size,d_model)
        self.pos_embed=nn.Embedding.from_pretrained(Position_embedding(max_seq_len,d_model))
        self.dropout_emb=nn.Dropout(dropout_emb)

    def forward(self,
                dec_input_ids:torch.Tensor,
                enc_input,
                mask_dec_attn,
                mask_cross_attn):
        dec_input=self.tgt_emb(dec_input_ids)
        seq_len=dec_input.size(1)
        dec_input=dec_input+self.pos_embed(torch.arange(0,seq_len,device=dec_input.device))
        dec_input=self.dropout_emb(dec_input)
        for layer in self.layers:
            dec_input=layer(dec_input,enc_input,mask_dec_attn,mask_cross_attn)
        return dec_input


class Transformer(nn.Module):
    def __init__(self,
                 frontend:nn.Module,
                 encoder:Encoder,
                 decoder:Decoder,
                 d_model:int,
                 vocab_size:int
                 ):
        super().__init__()
        self.frontend=frontend
        self.encoder=encoder
        self.decoder=decoder
        self.LM_head=nn.Linear(d_model,vocab_size)

    def forward(self,
                enc_input:torch.Tensor,
                enc_lens:torch.Tensor,
                dec_lens:torch.Tensor,
                dec_input_ids:torch.Tensor,
                ):
        enc_input=self.frontend(enc_input)
        batch_size=enc_input.size(0)
        max_enc_len=enc_input.size(1)
        enc_mask=get_enc_mask(batch_size,max_enc_len,enc_lens,device=enc_input.device)
        enc_out=self.Encoder(enc_input,enc_mask)

        max_dec_len=dec_input_ids.size(1)
        dec_mask=get_dec_mask(batch_size,max_dec_len,dec_lens,dec_input_ids.device)
        dec_enc_mask=get_dec_enc_mask(batch_size,max_enc_len,enc_lens,max_dec_len,dec_input_ids.device)
        dec_out=self.decoder(dec_input_ids,enc_out,dec_mask,dec_enc_mask)

        return self.LM_head(dec_out)

