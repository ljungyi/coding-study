import torch
from transformer import get_dec_mask,get_enc_mask,get_dec_enc_mask

def generate(model,device,src_input_ids,src_lens,
             max_new_tokens,
             bos,pad,eos,
             use_KV_cache:bool=False,
             max_seq_len:int=None,
             d_head:int=None,num_layers:int=None,
             num_heads:int=None):

    batch_size=src_input_ids.size(0)
    enc_max_len=src_input_ids.size(1)
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
    
    if use_KV_cache:
        KV_cache={
            "K_cache":torch.zeros([num_layers,batch_size,num_heads,max_seq_len,d_head],
                                  device=device),
            "V_cache":torch.zeros([num_layers,batch_size,num_heads,max_seq_len,d_head],
                                  device=device),
            "length" :0
        }
        gen_enc_mask=get_dec_enc_mask(batch_size,enc_max_len,src_lens,
                                1,device)
    with torch.no_grad():
        if not use_KV_cache:
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
                # finished=finished | (next_token==eos)
                # if finished.all():
                #     break
        else:
            for i in range(max_new_tokens):
                if i==0:
                    gen_hidden_states,KV_cache=model.decoder(generated,enc_out,None,gen_enc_mask,
                                                            use_KV_cache,KV_cache)
                else:
                    gen_hidden_states,KV_cache=model.decoder(next_token.unsqueeze(1),enc_out,None,gen_enc_mask,
                                                    use_KV_cache,KV_cache
                                                    )
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
                # finished=finished | (next_token==eos)
                # if finished.all():
                #     break
    return generated