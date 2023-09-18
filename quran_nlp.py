import torch
from re import sub
import math
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

with open('quran-final.txt') as f:
    s = f.read()

s = s.replace('\n', ' ')

q = sub('[^ةجحخهعغإفقثصضشسيىبلاآتنمكوؤرزأدءذئ طظ]', '', s)

q_characters = list(q)  # quran characters

q_words = q.split(' ')  # quran words

len(set(q_characters))    # 36 including the space

chr_int_dict = {l:i for (i, l) in enumerate(set(q_characters), start=1)}  # dictionary of keys -> quran characters and values -> intgers

q_index = [chr_int_dict[chr] for chr in q_characters] # quran characters to intgers


def get_loader(data , num_step=5):
    arr = torch.tensor([data[i:i+num_step + 1] for i in range(0, len(data)-num_step, num_step)])
    for i in range(len(arr)):
        yield arr[i][:num_step].view(1,-1) , arr[i][1:num_step+1].view(1,-1)


#-------------------------------------------------#

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from re import sub
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('quran-final.txt') as f:
    text = f.read()

tokenizer = get_tokenizer(lambda x: list(sub('[^ةجحخهعغإفقثصضشسيىبلاآتنمكوؤرزأدءذئ طظ]', '', text)))

vocab = build_vocab_from_iterator(iter(tokenizer(text)))  # the vocabulary which here is the characters

class Quran(torch.utils.data.Dataset):
    def __init__(self, text, n_steps=5):
        self.text = tokenizer(text)
        self.n_steps = n_steps
    def __getitem__(self, i):
        return self.text[i:i+self.n_steps+1]
    def __len__(self):
        return len(self.text) - self.n_steps

q_dataset = Quran(text)

def collat(batch):
    X , Y = [], []
    for b in batch:
        X.append(torch.tensor(vocab(b)[:-1]))
        Y.append(torch.tensor(vocab(b)[1:]))
    return X, Y

dataloader = DataLoader(q_dataset, batch_size=8, shuffle=False, collate_fn=collat)

next(iter(dataloader))

