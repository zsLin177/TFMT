import torch
import torch.nn as nn
from SubLayers import Mutlti_Head_Attention,PositionwiseFeedForward

class Encoder_layer(nn.Module):
    def __init__(self,d_model,d_inner,n_heads,d_k,d_v,dropout):
        super().__init__()
        self.slf_attention=Mutlti_Head_Attention(n_head=n_heads,d_model=d_model,d_k=d_k,
                                                 d_v=d_v,dropout=dropout)
        self.fw=PositionwiseFeedForward(d_in=d_model,d_hid=d_inner,dropout=dropout)

    def forward(self,input,slf_attn_mask=None):
        output=self.slf_attention(input,input,input,slf_attn_mask)
        output=self.fw(output)
        return output

class Decoder_layer(nn.Module):
    def __init__(self,d_model,d_inner,n_heads,d_k,d_v,dropout):
        super().__init__()
        self.slf_attn = Mutlti_Head_Attention(n_head=n_heads,d_model=d_model,d_k=d_k,
                                                 d_v=d_v,dropout=dropout)
        self.enc_attn = Mutlti_Head_Attention(n_head=n_heads,d_model=d_model,d_k=d_k,
                                                 d_v=d_v,dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_in=d_model,d_hid=d_inner,dropout=dropout)

    def forward(self,dec_input,enc_output,slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output=self.slf_attn(dec_input,dec_input,dec_input,slf_attn_mask)
        dec_output=self.enc_attn(dec_output,enc_output,enc_output,dec_enc_attn_mask)
        dec_output=self.pos_ffn(dec_output)
        return dec_output
