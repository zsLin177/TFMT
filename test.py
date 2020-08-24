import json

max_len=0
with open('./data/train.json','r') as f:
    data=json.load(f)
    for d in data:
        max_len=max(max_len,len(d['question_toks']))
    print(max_len)