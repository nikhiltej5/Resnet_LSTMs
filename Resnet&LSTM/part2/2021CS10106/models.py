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

special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>","<num_value>","<str_value>"]
# keywords = [ '(', ')', 'add','subtract','divide','power','multiply', 'negate','circle_area','square_area', 'rectangle_area', 'volume_cylinder','const_', 'floor','sqrt','circumface','inverse', 'factorial', 'log', 'choose', 'rhombus_perimeter', 'rectangle_perimeter','quadrilateral_area','speed','reminder','volume_rectangular_prism', 'permutation','surface_sphere','triangle_area','gcd','lcm','triangle_perimeter','p_after_gain','triangle_area_three_edges','cube_edge_by_volume', 'surface_rectangular_prism', 'square_perimeter', 'max','surface_cube','volume_cube','rhombus_area','original_price_before_loss', 'square_edge_by_area','volume_cone', 'stream_speed','surface_cylinder', 'volume_sphere', 'original_price_before_gain','min','square_edge_by_perimeter', 'negate_prob']

class GloveEmbedding:
    def __init__(self,embed_dim, word2idx):
        self.embed_dim = embed_dim
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)

    def get_embed_matrix(self):
        glove = GloVe(name='6B', dim=self.embed_dim)
        # glove = torch.load("/processing/glove_embeddings.pt")
        embedding_matrix = torch.zeros(self.vocab_size, self.embed_dim)
        for word, idx in self.word2idx.items():
            if word in special_tokens:
                embedding_matrix[idx] = torch.randn(self.embed_dim)
            elif word in glove.stoi:
                embedding_matrix[idx] = torch.tensor(glove.vectors[glove.stoi[word]])
            else:
                embedding_matrix[idx] = embedding_matrix[3]
        return embedding_matrix

class LSTMencoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units, num_layers, bidirectional=True, p = 0.5, embed_matrix=None):
        super(LSTMencoder, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.bidirectional = bidirectional
        self.embed_matrix = embed_matrix
        if self.embed_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_matrix, padding_idx=0)
        else:
            self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(embed_dim, hidden_units, num_layers = num_layers, dropout=p, batch_first=True, bidirectional=self.bidirectional)
        self.hidden = nn.Linear(2*hidden_units, hidden_units)
        self.cell = nn.Linear(2*hidden_units, hidden_units)

    def forward(self, x):
        if (self.num_layers > 1):
            x = self.dropout(self.embedding(x))
        else:
            x = self.embedding(x)
        output, (hidden, cell) = self.LSTM(x)
        hidden = self.hidden(torch.cat((hidden[0:1],hidden[1:2]),dim=2))
        cell = self.cell(torch.cat((cell[0:1],cell[1:2]),dim=2))

        return output, hidden, cell
    
class LSTMDecoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units, num_layers, p = 0.5, embed_matrix=None):
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embed_matrix = embed_matrix
        if self.embed_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_matrix, padding_idx=0)
        else:
            self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(embed_dim, hidden_units, num_layers=num_layers, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_units, input_size)

    def forward(self, x, h0_c0):
        if (self.num_layers > 1):
            x = self.dropout(self.embedding(x))
        else:
            x = self.embedding(x)
        x = x.unsqueeze(1)
        output, (hidden, cell) = self.LSTM(x, h0_c0)
        print(output.shape)
        out = self.fc(output)

        return out, (hidden, cell)
    
class Seq2Seq(nn.Module):
    def __init__(self, embed_dim=200,en_hidden=512, de_hidden=512,en_num_layers=1,de_num_layers=1):
        super(Seq2Seq, self).__init__()
        self.embed_dim = embed_dim
        self.en_hidden = en_hidden
        self.de_hidden = de_hidden
        self.en_num_layers = en_num_layers
        self.de_num_layers = de_num_layers
        with open("/processing/encoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.en_word2idx = word2idx
        with open("/processing/decoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.de_word2idx = word2idx
        self.embed_matrix = GloveEmbedding(embed_dim, self.en_word2idx).get_embed_matrix()
        self.encoder = LSTMencoder(len(self.en_word2idx), embed_dim, en_hidden, en_num_layers, embed_matrix=self.embed_matrix)
        self.decoder = LSTMDecoder(len(self.de_word2idx), embed_dim, de_hidden, de_num_layers, embed_matrix=self.embed_matrix)
        self.start_idx = self.en_word2idx["<sos>"]
        self.end_idx = self.en_word2idx["<eos>"]

    def forward(self, Problem, Linear_formula, teacher_forcing_ratio=0.6):
        batch_size = Problem.size(0)
        max_len = Linear_formula.size(1)
        
        _, hidden, cell = self.encoder(Problem)
        
        outputs = torch.zeros(batch_size, max_len, len(self.de_word2idx)).to(device)
        
        x = Linear_formula[:,0]
        
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(x, (hidden, cell))
            outputs[:,t,:] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            x = Linear_formula[:,t] if teacher_force else output.argmax(2).squeeze()

        return outputs, Linear_formula
    
class AttentionLayers(nn.Module):
    def __init__(self, hidden_units):
        super(AttentionLayers, self).__init__()
        self.hidden_units = hidden_units
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attn = nn.Linear(hidden_units*3, hidden_units)
        self.v = nn.Linear(hidden_units, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.repeat(encoder_outputs.size(1),1,1).transpose(0,1)
        energy = self.relu(self.attn(torch.cat((hidden, encoder_outputs),dim=2)))
        attention = self.v(energy).squeeze(2)
        attention_weights = self.softmax(attention).unsqueeze(1)
        context_vector =  torch.bmm(attention_weights, encoder_outputs)
        return context_vector
    
class AttentionLayersBert(nn.Module):
    def __init__(self, hidden_units):
        super(AttentionLayersBert, self).__init__()
        self.hidden_units = hidden_units
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attn = nn.Linear(hidden_units*2, hidden_units)
        self.v = nn.Linear(hidden_units, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.repeat(encoder_outputs.size(1),1,1).transpose(0,1)
        energy = self.relu(self.attn(torch.cat((hidden, encoder_outputs),dim=2)))
        attention = self.v(energy).squeeze(2)
        attention_weights = self.softmax(attention).unsqueeze(1)
        context_vector =  torch.bmm(attention_weights, encoder_outputs)
        return context_vector,attention_weights
    
class LSTMdecoderAttention(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units, num_layers, p = 0.5, embed_matrix=None,bidirectional=False):
        super(LSTMdecoderAttention, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embed_matrix = embed_matrix
        self.bidirectional = bidirectional
        if self.embed_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_matrix, padding_idx=0)
        else:
            self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(2*hidden_units +embed_dim, hidden_units, num_layers=num_layers, dropout=p, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_units, input_size)
        self.attention = AttentionLayers(hidden_units)
    
    def forward(self, x, h0_c0, encoder_outputs):
        if (self.num_layers > 1):
            x = self.dropout(self.embedding(x))
        else:
            x = self.embedding(x)
        x = x.unsqueeze(1)
        context_vector = self.attention(h0_c0[0], encoder_outputs)
        x = torch.cat((context_vector, x), dim=2)
        output, (hidden, cell) = self.LSTM(x, h0_c0)
        out = self.fc(output)
        
        return out, (hidden, cell)
    
class Seq2SeqAttention(nn.Module):
    def __init__(self,embed_dim=200,en_hidden=512, de_hidden=512,en_num_layers=1,de_num_layers=1):
        super(Seq2SeqAttention, self).__init__()
        self.embed_dim = embed_dim
        self.en_hidden = en_hidden
        self.de_hidden = de_hidden
        self.en_num_layers = en_num_layers
        self.de_num_layers = de_num_layers
        with open("/processing/encoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.en_word2idx = word2idx
        with open("/processing/decoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.de_word2idx = word2idx
        self.embed_matrix = GloveEmbedding(embed_dim, self.en_word2idx).get_embed_matrix()
        self.encoder = LSTMencoder(len(self.en_word2idx), embed_dim, en_hidden, en_num_layers, embed_matrix=self.embed_matrix)
        self.decoder = LSTMdecoderAttention(len(self.de_word2idx), embed_dim, de_hidden, de_num_layers, embed_matrix=self.embed_matrix)
        self.start_idx = self.en_word2idx["<sos>"]
        self.end_idx = self.en_word2idx["<eos>"] 

    def forward(self, Problem, Linear_formula, teacher_forcing_ratio=0.6):
        batch_size = Problem.size(0)
        max_len = Linear_formula.size(1)
        
        encoder_outputs, hidden, cell = self.encoder(Problem)
        
        outputs = torch.zeros(batch_size, max_len, len(self.de_word2idx)).to(device)
        
        x = Linear_formula[:,0]
        
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(x, (hidden, cell), encoder_outputs)
            outputs[:,t,:] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            x = Linear_formula[:,t] if teacher_force else output.argmax(2).squeeze()

        return outputs, Linear_formula
    
#bert encoder frozen 
class BertEncoder(nn.Module):
    def __init__(self, hidden_units, num_layers, bidirectional=True):
        super(BertEncoder, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x, attention_mask):
        output = self.bert(x, attention_mask)
        hidden = output.last_hidden_state
        return hidden
    
class BerttunedEncoder(nn.Module):
    def __init__(self, hidden_units, num_layers, bidirectional=True):
        super(BerttunedEncoder, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bert = BertModel.from_pretrained('bert-base-uncased')
#         for param in self.bert.parameters():
#             param.requires_grad = True

    def forward(self, x, attention_mask):
        output = self.bert(x, attention_mask)
        output = output.last_hidden_state
        return output

class LSTMdecoderAttentionBert(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units, num_layers, p = 0.5, embed_matrix=None,bidirectional=False):
        super(LSTMdecoderAttentionBert, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embed_matrix = embed_matrix
        self.bidirectional = bidirectional
        if self.embed_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_matrix, padding_idx=0)
        else:
            self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(hidden_units +embed_dim, hidden_units, num_layers=num_layers, dropout=p, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_units, input_size)
        self.attention = AttentionLayersBert(hidden_units)
    
    def forward(self, x, h0_c0, encoder_outputs):
        if (self.num_layers > 1):
            x = self.dropout(self.embedding(x))
        else:
            x = self.embedding(x)
        x = x.unsqueeze(1)
        context_vector,_ = self.attention(h0_c0[0], encoder_outputs)
        x = torch.cat((context_vector, x), dim=2)
        output, (hidden, cell) = self.LSTM(x, h0_c0)
        out = self.fc(output)
        
        return out, (hidden, cell)
    
class Bert2SeqAttention(nn.Module):
    def __init__(self,embed_dim=200,en_hidden=768, de_hidden=768,en_num_layers=1,de_num_layers=1):
        super(Bert2SeqAttention, self).__init__()
        self.embed_dim = embed_dim
        self.en_hidden = en_hidden
        self.de_hidden = de_hidden
        self.en_num_layers = en_num_layers
        self.de_num_layers = de_num_layers
        with open("/processing/encoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.en_word2idx = word2idx
        with open("/processing/decoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.de_word2idx = word2idx
        self.embed_matrix = None
        self.encoder = BertEncoder(en_hidden, en_num_layers)
        self.decoder = LSTMdecoderAttentionBert(len(self.de_word2idx), embed_dim, de_hidden, de_num_layers, embed_matrix=self.embed_matrix)
        self.start_idx = self.en_word2idx["<sos>"]
        self.end_idx = self.en_word2idx["<eos>"] 

    def forward(self, Problem,attention_mask, Linear_formula, teacher_forcing_ratio=0.6):
        batch_size = Problem.size(0)
        max_len = Linear_formula.size(1)
        
        encoder_outputs = self.encoder(Problem, attention_mask)
        
        outputs = torch.zeros(batch_size, max_len, len(self.de_word2idx)).to(device)
        
        x = Linear_formula[:,0]
        hidden = torch.zeros(1, batch_size, self.de_hidden).to(device)
        cell = torch.zeros(1, batch_size, self.de_hidden).to(device)
        
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(x, (hidden, cell), encoder_outputs)
            outputs[:,t,:] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            x = Linear_formula[:,t] if teacher_force else output.argmax(2).squeeze()

        return outputs, Linear_formula
    
class Bert2SeqAttentionTuned(nn.Module):
    def __init__(self,embed_dim=200,en_hidden=768, de_hidden=768,en_num_layers=1,de_num_layers=1):
        super(Bert2SeqAttentionTuned, self).__init__()
        self.embed_dim = embed_dim
        self.en_hidden = en_hidden
        self.de_hidden = de_hidden
        self.en_num_layers = en_num_layers
        self.de_num_layers = de_num_layers
        with open("/processing/encoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.en_word2idx = word2idx
        with open("/processing/decoder_word2idx.pickle", "rb") as file:
            word2idx = pickle.load(file)
        self.de_word2idx = word2idx
        self.embed_matrix = None
        self.encoder = BerttunedEncoder(en_hidden, en_num_layers)
        self.decoder = LSTMdecoderAttentionBert(len(self.de_word2idx), embed_dim, de_hidden, de_num_layers, embed_matrix=self.embed_matrix)
        self.start_idx = self.en_word2idx["<sos>"]
        self.end_idx = self.en_word2idx["<eos>"] 

    def forward(self, Problem,attention_mask, Linear_formula, teacher_forcing_ratio=0.6):
        batch_size = Problem.size(0)
        max_len = Linear_formula.size(1)
        
        encoder_outputs = self.encoder(Problem, attention_mask)
        
        outputs = torch.zeros(batch_size, max_len, len(self.de_word2idx)).to(device)
        
        x = Linear_formula[:,0]
        hidden = torch.zeros(1, batch_size, self.de_hidden).to(device)
        cell = torch.zeros(1, batch_size, self.de_hidden).to(device)
        
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(x, (hidden, cell), encoder_outputs)
            outputs[:,t,:] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            x = Linear_formula[:,t] if teacher_force else output.argmax(2).squeeze()

        return outputs, Linear_formula
        