import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from collections import Counter
import gc
import random
from utils import *
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def beam_search(decoder, hidden, cell, start_idx, max_len, k):
    sequences = [([start_idx],(hidden,cell), 1)]
    i = 0
    while i < max_len-1:
        new_sequences = []
        for seq, (hidden, cell), score in sequences:
            x = torch.LongTensor([seq[-1]]).to(device)
            output, (hidden, cell) = decoder(x, (hidden, cell))
            output = output.squeeze(1)
            topv,topi = output.topk(k,dim=1)
            for j in range(k):
                new_idx = topi[0][j].item()
                new_seq = seq + [new_idx]
                new_prob = torch.log(topv[0][j])
                updated_score = score - new_prob
                new_candidate = (new_seq, (hidden, cell), updated_score)
                new_sequences.append(new_candidate)
        new_sequences.sort(key=lambda x: x[2])
        sequences = new_sequences[:k]
        i += 1
    best = sequences[0][0]
    words = torch.zeros(1,max_len, dtype=torch.long)
    for t in range(max_len):
        words[:,t] = torch.LongTensor([best[t]])
    return words

def beam_search_attn(decoder, hidden, cell,start_idx,end_idx, max_len,k,en_out):
    sequences = [([start_idx],(hidden,cell), 1)]
    i = 0
    while i < max_len-1:
        new_sequences = []
        for seq, (hidden, cell), score in sequences:
            x = torch.LongTensor([seq[-1]]).to(device)
            output, (hidden, cell) = decoder(x, (hidden, cell),en_out)
            output = output.squeeze(1)
            topv,topi = output.topk(k,dim=1)
            for j in range(k):
                new_idx = topi[0][j].item()
                new_seq = seq + [new_idx]
                new_prob = torch.log(topv[0][j])
                updated_score = score - new_prob
                new_candidate = (new_seq, (hidden, cell), updated_score)
                new_sequences.append(new_candidate)
        new_sequences.sort(key=lambda x: x[2])
        sequences = new_sequences[:k]
        i += 1
    best = sequences[0][0]
    words = torch.zeros(1,max_len, dtype=torch.long)
    for t in range(max_len):
        words[:,t] = torch.LongTensor([best[t]])
    return words