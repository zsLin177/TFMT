import spacy
from torchtext import data, datasets

def DataLoader(device,test=False):
    '''
    :param test: 是否有测试集
    '''

    spacy_en = spacy.load('en')

    def tokenizer(text): # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<pad>"

    SRC = data.Field(tokenize=tokenizer, pad_token=BLANK_WORD,fix_length=25, batch_first=True)
    TGT = data.Field(tokenize=tokenizer, init_token = BOS_WORD,
                         eos_token = EOS_WORD, pad_token=BLANK_WORD, fix_length=25, batch_first=True)

    train,dev = data.TabularDataset.splits(path='./data', train='train.json', validation='dev.json',
                                           format='json',
                                           fields={"query":('tgt',TGT), "question":('src',SRC)})

    MIN_FREQ = 3
    SRC.build_vocab(train.src,min_freq = MIN_FREQ)
    TGT.build_vocab(train.tgt,min_freq = MIN_FREQ)
    n_src_words = len(SRC.vocab.stoi)
    n_tgt_words = len(TGT.vocab.stoi)
    src_pad_idx = SRC.vocab.stoi['<pad>']
    tgt_pad_idx = TGT.vocab.stoi['<pad>']

    train_iter = data.BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.src),
                                     shuffle=True,device=device)
    dev_iter = data.BucketIterator(dev, batch_size=32, sort_key=lambda x: len(x.src),
                                     shuffle=True,device=device)

    return train_iter, dev_iter, src_pad_idx,\
           tgt_pad_idx, n_src_words, n_tgt_words, \
           SRC.vocab, TGT.vocab


