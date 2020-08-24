import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self,scale,dropout=0.1):
        super().__init__()
        self.scale=scale
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):
        att=torch.matmul(q/self.scale,k.transpose(2,3))
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)

        att=self.dropout(F.softmax(att,dim = -1))
        output=torch.matmul(att,v)
        return output