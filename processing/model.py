# TODOs
# CUDA - GPUs

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

HIDDEN_STATE_SIZE = 512
EMBEDDING_DIM = 50

class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel, self).__init__()

        self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hid_size, dict_size)
        )

        

