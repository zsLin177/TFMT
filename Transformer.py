import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Layers import Encoder_layer,Decoder_layer
from supar.utils import Config

def get_pad_mask(seq, pad_idx):
    '''
    :param seq: [batch_size, seq_len]
    :param pad_idx: 0
    :return: [batch_size,1,seq_len]
    '''
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        # 放到内存，对于每次不同的输入句子，位置词向量是不需要改变的（不需要去学习改变的）
        # 之后使用就用的是self.pos_table

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        # 先整除2再乘以2，是因为对于奇数维也变成相应的偶数维

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # [:, 0::2]前一个：对所有的行，后面是对偶数列，抽出来构成一个新的array

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
        #unsqueeze表示增加一个维度（加一个[])，将参数作为一个整体

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
        # .clone.detach(),开辟一个新内存，复制内容，从计算图中抽出来，不计算梯度
        # x:[batch_size,seq_len,embed_size] pos_table:[1,200,embed_size]

class Encoder(nn.Module):
    def __init__(self,n_src_words,pad_idx,d_word_vec,d_model,d_inner,n_layers,
                 n_heads,d_k,d_v,dropout,n_positions=200):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_embed = nn.Embedding(n_src_words,d_word_vec,padding_idx=pad_idx)
        self.position_encoder = PositionalEncoding(d_word_vec,n_positions)
        self.layer_stack=nn.ModuleList([Encoder_layer(d_model,d_inner,n_heads,d_k,d_v,dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self,src_seq,src_mask):
        enc_output = self.dropout(self.position_encoder(self.src_embed(src_seq)))
        enc_output = self.layer_norm(enc_output)
        for layer in self.layer_stack:
            enc_output = layer(enc_output, slf_attn_mask=src_mask)
        return enc_output

class Decoder(nn.Module):
    def __init__(self,n_tgt_words,pad_idx,d_word_vec,d_model,d_inner,n_layers,
                 n_heads,d_k,d_v,dropout,n_positions=200):
        super().__init__()
        self.pad_idx = pad_idx
        self.tgt_embed = self.src_embed = nn.Embedding(n_tgt_words,d_word_vec,padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_positions)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([Decoder_layer(d_model,d_inner,n_heads,d_k,d_v,dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt_seq, tgt_mask, enc_output, src_mask):
        dec_output = self.dropout(self.position_enc(self.tgt_embed(tgt_seq)))
        dec_output = self.layer_norm(dec_output)
        for layer in self.layer_stack:
            dec_output = layer(dec_output,enc_output,tgt_mask,src_mask)
        return dec_output


class Transformer(nn.Module):
    def __init__(self,n_src_words,n_tgt_words,src_pad_idx,tgt_pad_idx,generator,
                 d_word_vec=512,d_model=512,d_inner=2048,
                 n_layers=6,n_heads=8,d_k=64,d_v=64,dropout=0.1,n_positions=200):
        super().__init__()
        self.args = self.args = Config().update(locals())
        self.d_model = d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, tgt_pad_idx
        self.encoder = Encoder(n_src_words,src_pad_idx,d_word_vec,d_model,d_inner,
                             n_layers,n_heads,d_k,d_v,dropout,n_positions)
        self.decoder = Decoder(n_tgt_words, tgt_pad_idx, d_word_vec, d_model,
                               d_inner, n_layers, n_heads, d_k, d_v, dropout, n_positions)
        self.generator = generator


    def forward(self, src_seq, tgt_seq, src_mask, tgt_mask):
        # src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        # tgt_mask = get_pad_mask(tgt_seq, self.trg_pad_idx) & get_subsequent_mask(tgt_seq)
        encoder_output = self.encoder(src_seq, src_mask)
        decoder_output = self.decoder(tgt_seq, tgt_mask, encoder_output, src_mask)
        return decoder_output

    @classmethod
    def load(cls, path):
        # 这个cls学到了
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device)
        model = cls(**state['args'])
        # 输入字典，初始化生成model
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict = self.state_dict()
        state = {
            'args': self.args,
            'state_dict': state_dict,
        }
        torch.save(state, path)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def make_model(src_vocab, tgt_vocab, src_pad_idx, tgt_pad_idx, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    model = Transformer(src_vocab,tgt_vocab,src_pad_idx,tgt_pad_idx, Generator(d_model,tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

if __name__ == '__main__':
    x=torch.randint(0,3,(2,3))
    print(x)
    pad_mask=get_pad_mask(x,0)
    print(pad_mask)
    print(pad_mask.shape)
    sub_mask=get_subsequent_mask(x)
    print(sub_mask)
    print(pad_mask & sub_mask)