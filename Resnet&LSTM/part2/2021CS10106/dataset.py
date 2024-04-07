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

def collate_fn(batch):
    max_len_Problem = max([len(x['Problem']) for x in batch])
    max_len_Linear_formula = max([len(x['Linear_formula']) for x in batch])

    Problem_lens = torch.zeros(len(batch), dtype=torch.long)
    Linear_formula_lens = torch.zeros(len(batch), dtype=torch.long)

    Padding_Problem = torch.zeros(len(batch), max_len_Problem, dtype=torch.long)
    Padding_Linear_formula = torch.zeros(len(batch), max_len_Linear_formula, dtype=torch.long)
    
    Answer = torch.zeros(len(batch))

    for i, data in enumerate(batch):
        Problem = data['Problem']
        Linear_formula = data['Linear_formula']
        Ans = data["answer"]
        Problem_lens[i] = len(Problem)
        Linear_formula_lens[i] = len(Linear_formula)

        Padding_Problem[i,:len(Problem)] = torch.LongTensor(Problem)
        Padding_Linear_formula[i,:len(Linear_formula)] = torch.LongTensor(Linear_formula)
        Answer[i] = Ans

    return {"Problem":Padding_Problem, "Linear_formula":Padding_Linear_formula, "Problem_lens":Problem_lens,"Linear_formula_lens":Linear_formula_lens,"answer":Answer}

def collate_fn_bert(batch):
    max_len_Problem = max([len(x['Problem']) for x in batch])
    max_len_Linear_formula = max([len(x['Linear_formula']) for x in batch])

    Problem_lens = torch.zeros(len(batch), dtype=torch.long)
    Linear_formula_lens = torch.zeros(len(batch), dtype=torch.long)

    Problem_mask = torch.zeros(len(batch), max_len_Problem, dtype=torch.long)
    Padding_Problem = torch.zeros(len(batch), max_len_Problem, dtype=torch.long)
    Padding_Linear_formula = torch.zeros(len(batch), max_len_Linear_formula, dtype=torch.long)
    Answer = torch.zeros(len(batch))

    for i, data in enumerate(batch):
        Problem = data['Problem']
        Linear_formula = data['Linear_formula']
        Ans = data["answer"]
        Problem_lens[i] = len(Problem)
        Linear_formula_lens[i] = len(Linear_formula)
        Problem_mask[i,:len(Problem)] = torch.ones((1,len(Problem)),dtype=torch.long)

        Padding_Problem[i,:len(Problem)] = torch.LongTensor(Problem)
        Padding_Linear_formula[i,:len(Linear_formula)] = torch.LongTensor(Linear_formula)
        Answer[i] = Ans

    return {"Problem":Padding_Problem, "Linear_formula":Padding_Linear_formula, "Problem_lens":Problem_lens,"Linear_formula_lens":Linear_formula_lens,"Problem_mask":Problem_mask,"answer":Answer}

class CustomDataset(Dataset):
    def __init__(self,path):
        self.data = pd.read_json(path)

        with open("/processing/encoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.encoder_word2idx = word2idx

        with open("/processing/decoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.decoder_word2idx = word2idx

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        Problem = ["<sos>"] + tokenize_text(self.data['Problem'][idx]) + ["<eos>"]
        Problem = [self.encoder_word2idx.get(x,3) for x in Problem]

        Linear_formula = ["<sos>"] + tokenize_linear_formula(self.data['linear_formula'][idx]) + ["<eos>"]
        Linear_formula = [self.decoder_word2idx.get(x,3) for x in Linear_formula]

        answer = self.data['answer'][idx]

        return {"Problem":Problem, "Linear_formula":Linear_formula, "answer":answer} 
    
class CustomDatasetBert(Dataset):
    def __init__(self,path):
        self.data = pd.read_json(path)

        with open("/processing/encoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.encoder_word2idx = word2idx

        with open("/processing/decoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.decoder_word2idx = word2idx
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        Problem = self.data['Problem'][idx]
        Problem = ["<sos>"] + tokenize_text(self.data['Problem'][idx]) + ["<eos>"]

        Linear_formula = ["<sos>"] + tokenize_linear_formula(self.data['linear_formula'][idx]) + ["<eos>"]
        Linear_formula = [self.decoder_word2idx.get(x,3) for x in Linear_formula]

        answer = self.data['answer'][idx]

        return {"Problem":Problem, "Linear_formula":Linear_formula, "answer":answer}
    
