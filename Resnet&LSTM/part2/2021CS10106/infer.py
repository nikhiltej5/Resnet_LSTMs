import json
from collections import Counter, defaultdict
import pickle
import pandas as pd
import subprocess
import argparse
import os
import json
import pickle
import shutil
from time import time
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from models import *
from dataset import *
from utils import *
from beamsearch import *
from evaluator import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(outputs, Problem, Answer, Linear_formula, de_idx2word, en_idx2word, op=0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    output_new = []
    Problem_new = []
    Linear_formula_new = []
    batch_size = Problem.size(0)
    dl = []
    for i in range(batch_size):
        t = []
        for x in outputs[i]:
            if (x.item() == 2):
                t.append(x.item())
                break
            t.append(x.item())
        output_new.append(t)
    for i in range(batch_size):
        t = []
        for x in Problem[i]:
            if (op == 0):
                if (x.item() == 2):
                    t.append(x.item())
                    break
                t.append(x.item())
            else:
                if (x.item() == tokenizer.encode("[SEP]")):
                    t.append(x.item())
                    break
                t.append(x.item())
        Problem_new.append(t)
    for i in range(batch_size):
        t = []
        for x in Linear_formula[i]:
            if (x.item() == 2):
                t.append(x.item())
                break
            t.append(x.item())
        Linear_formula_new.append(t)
    #add into dataframe df
    for i in range(batch_size):
        if (op == 0):
            dl.append({"Problem": " ".join([en_idx2word[x] for x in Problem_new[i][1:-1]]), "Predicted": "".join([de_idx2word[x] for x in output_new[i][1:-1]]), "linear_formula": "".join([de_idx2word[x] for x in Linear_formula_new[i][1:-1]]), "answer": Answer[i].item()})
        else:
            dl.append({"Problem": tokenizer.decode(Problem_new[i][1:-1]), "Predicted": "".join([de_idx2word[x] for x in output_new[i][1:-1]]), "linear_formula": "".join([de_idx2word[x] for x in Linear_formula_new[i][1:-1]]), "answer": Answer[i].item()})
    return dl

def eval_model(args):
    dl = []
    if args.model_type == "bert_lstm_attn_frozen":
        model = Bert2SeqAttentionTuned().to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_file))
        with open("/processing/decoder_idx2word.pickle", "rb") as file:
            de_idx2word = pickle.load(file)
        with open("/processing/encoder_idx2word.pickle", "rb") as file:
            en_idx2word = pickle.load(file)    
        val_dataset = CustomDatasetBert(args.test_data_file)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=True, collate_fn=collate_fn_bert)
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(val_loader):
                Problem = data['Problem'].to(device)
                Linear_formula = data['Linear_formula'].to(device)
                attention_mask = data['Problem_mask'].to(device)
                Answer = data['answer']
                max_len = Linear_formula.size(1)
                outputs = model.module.encoder(Problem, attention_mask)
                hidden = torch.zeros(1, Problem.size(0), 768).to(device)
                cell = torch.zeros(1, Problem.size(0), 768).to(device)
                words = torch.zeros(Problem.size(0), max_len, dtype=torch.long)
                for j in range(Problem.size(0)):
                    words[j,:] = beam_search_attn(model.module.decoder,hidden[:,j,:].unsqueeze(1),cell[:,j,:].unsqueeze(1),1,2,max_len,args.beam_size,outputs[j,:,:].unsqueeze(0))
                outputs = words
                dl += evaluate(outputs,Problem, Answer, Linear_formula, de_idx2word, en_idx2word, 1)
    elif args.model_type == "bert_lstm_attn_tuned":
        model = Bert2SeqAttentionTuned().to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_file))
        with open("/processing/decoder_idx2word.pickle", "rb") as file:
            de_idx2word = pickle.load(file)
        with open("/processing/encoder_idx2word.pickle", "rb") as file:
            en_idx2word = pickle.load(file)
        val_dataset = CustomDatasetBert(args.test_data_file)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=True, collate_fn=collate_fn_bert)
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(val_loader):
                Problem = data['Problem'].to(device)
                Linear_formula = data['Linear_formula'].to(device)
                attention_mask = data['Problem_mask'].to(device)
                Answer = data['answer']
                max_len = Linear_formula.size(1)
                outputs = model.module.encoder(Problem, attention_mask)
                hidden = torch.zeros(1, Problem.size(0), 768).to(device)
                cell = torch.zeros(1, Problem.size(0), 768).to(device)
                words = torch.zeros(Problem.size(0), max_len, dtype=torch.long)
                for j in range(Problem.size(0)):
                    words[j,:] = beam_search_attn(model.module.decoder,hidden[:,j,:].unsqueeze(1),cell[:,j,:].unsqueeze(1),1,2,max_len,args.beam_size,outputs[j,:,:].unsqueeze(0))
                outputs = words
                dl += evaluate(outputs,Problem, Answer, Linear_formula, de_idx2word, en_idx2word, 1)
    elif args.model_type == "lstm_lstm_attn":
        model = Seq2SeqAttention().to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_file))
        with open("/processing/decoder_idx2word.pickle", "rb") as file:
            de_idx2word = pickle.load(file)
        with open("/processing/encoder_idx2word.pickle", "rb") as file:
            en_idx2word = pickle.load(file)
        val_dataset = CustomDataset(args.test_data_file)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=True, collate_fn=collate_fn)
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(val_loader):
                Problem = data['Problem'].to(device)
                Linear_formula = data['Linear_formula'].to(device)
                Answer = data['answer']
                max_len = Linear_formula.size(1)
                outputs,hidden,cell = model.module.encoder(Problem)
                words = torch.zeros(Problem.size(0), max_len, dtype=torch.long)
                for j in range(Problem.size(0)):
                    words[j,:] = beam_search_attn(model.module.decoder,hidden[:,j,:].unsqueeze(1),cell[:,j,:].unsqueeze(1),1,2,max_len,args.beam_size,outputs[j,:,:].unsqueeze(0))
                outputs = words
                dl += evaluate(outputs,Problem, Answer, Linear_formula, de_idx2word, en_idx2word)
    else:
        model = Seq2Seq().to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_file))
        with open("/processing/decoder_idx2word.pickle", "rb") as file:
            de_idx2word = pickle.load(file)
        with open("/processing/encoder_idx2word.pickle", "rb") as file:
            en_idx2word = pickle.load(file)
        val_dataset = CustomDataset(args.test_data_file)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=True, collate_fn=collate_fn)
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(val_loader):
                Problem = data['Problem'].to(device)
                Linear_formula = data['Linear_formula'].to(device)
                Answer = data['answer']
                max_len = Linear_formula.size(1)
                outputs,hidden,cell = model.module.encoder(Problem)
                words = torch.zeros(Problem.size(0), max_len, dtype=torch.long)
                for j in range(Problem.size(0)):
                    words[j,:] = beam_search(model.module.decoder,hidden[:,j,:].unsqueeze(1),cell[:,j,:].unsqueeze(1),1,2,max_len,args.beam_size)
                outputs = words
                dl += evaluate(outputs,Problem, Answer, Linear_formula, de_idx2word, en_idx2word)
    #convert some columns of df to json and save
    df = pd.DataFrame(dl)
    df.to_json(args.test_data_file, orient="records", indent=4)     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to evaluate")
    parser.add_argument("--model_file", type=str, required=True, help="Path to model file")
    parser.add_argument("--test_data_file", type=str, required=True, help="Path to test data file")
    parser.add_argument("--search", type=str, required=True, help="Search method")
    parser.add_argument("--beam_size", type=int, required=True, help="Beam size")
    args = parser.parse_args()
    eval_model(args)
