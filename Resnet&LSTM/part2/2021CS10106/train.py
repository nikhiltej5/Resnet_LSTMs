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
from transformers import BertModel, BertTokenizer

from utils import *
from beamsearch import * 
from dataset import *
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Bert2SeqAttentionTuned().to(device)
model = nn.DataParallel(model)
# model.load_state_dict(torch.load('/kaggle/working/my_model_34.pth'))

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

train_dataset = CustomDatasetBert("data/train.json")
train_loader = DataLoader(train_dataset, batch_size=32,num_workers=4, shuffle=True, collate_fn=collate_fn_bert)

val_dataset = CustomDatasetBert("data/dev.json")
val_loader = DataLoader(val_dataset, batch_size=32,num_workers=4, shuffle=True, collate_fn=collate_fn_bert)

train_size = len(train_loader)
model.train()
with open("/kaggle/working/seq2seq.txt", "w") as file:
    for epoch in range(1):
        for i,data in enumerate(train_loader):
            Problem = data['Problem'].to(device)
            Linear_formula = data['Linear_formula'].to(device)
            attention_mask = data['Problem_mask'].to(device)
            optimizer.zero_grad()
            output,_ = model(Problem, attention_mask, Linear_formula)
            output = output.reshape(-1, output.size(2))
            target = Linear_formula.reshape(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
                file.write(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}\n")

        scheduler.step()
        torch.save(model.state_dict(), "seq2seq.pth")

#generate loss curves using losses.txt,losses_0.3.txt,losses_0.9.txt in rbg colors files in matplotlib
import matplotlib.pyplot as plt
import numpy as np
with open("/kaggle/input/losses/losses.txt", "r") as file:
    data = file.readlines()
    data = [float(x.split()[-1]) for x in data]
    plt.plot(np.arange(1,len(data)+1),data,label="Teacher Forcing Ratio: 0.6")
with open("/kaggle/input/losses/losses_0.3.txt", "r") as file:
    data = file.readlines()
    data = [float(x.split()[-1]) for x in data]
    plt.plot(np.arange(1,len(data)+1),data,label="Teacher Forcing Ratio: 0.3")
with open("/kaggle/input/losses/losses_0.9.txt", "r") as file:
    data = file.readlines()
    data = [float(x.split()[-1]) for x in data]
    plt.plot(np.arange(1,len(data)+1),data,label="Teacher Forcing Ratio: 0.9")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

import matplotlib.pyplot as plt
with open("/kaggle/input/models1/seq2seqAttn/losses.txt", "r") as file:
    lines = file.readlines()
    epoch = [int(x.split()[1].split('/')[0]) for x in lines]
    losses = [float(x.split()[-1]) for x in lines]
    plt.plot(epoch, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()