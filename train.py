import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from Transformer import make_model
from Load_data import DataLoader
from Opt import NoamOpt

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


def run_epoch(data_iter, model, loss_compute, pad_idx):
    "Standard Training and Logging Function"
    model.train()
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        batch = Batch(batch.src, batch.tgt, pad_idx)
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def train(model, data_iter, loss_com, pad_idx, epochs=300, patenice=10):
    min_loss = float('inf')
    min_epoch = -1
    for epoch in range(epochs):
        start = datetime.now()
        loss = run_epoch(data_iter,model,loss_com,pad_idx)
        print('epoch %d:\nloss:%.5f'%(epoch+1,loss))
        t = datetime.now() - start
        if(loss<min_loss):
            min_loss = loss
            min_epoch = epoch
            model.save('./trans_model.pkl')
            print(f"{t}s elapsed (saved)\n")
        if((epoch - min_epoch) > patenice):
            print(f"{t}s elapsed\n")
            break
    print('train completed best model at epoch: %d ,loss:%.5f'%(min_epoch,min_loss))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_iter, dev_iter, src_pad_idx, tgt_pad_idx, n_src_words, n_tgt_words, src_vocab, tgt_vocab = \
    DataLoader(device)
    model = make_model(n_src_words, n_tgt_words, src_pad_idx, tgt_pad_idx)
    model = model.to(device)
    criterion = LabelSmoothing(size=len(tgt_vocab), padding_idx=tgt_pad_idx, smoothing=0.1)
    criterion.to(device)
    model_opt = NoamOpt(model.d_model, 1, 4000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    train(model, train_iter, SimpleLossCompute(model.generator,criterion,model_opt),tgt_pad_idx)