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

        # Init Embeddings Module | In: Vocab, Embedding Dimensions | Out:  Matrix Vocab*Embedding Dimensions
        self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        # Init LSTM Encoder
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        # Init LSTM Decoder
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        # Init Linear Output Transform to Logits (not Softmax output yet)
        self.output = nn.Sequential(
            nn.Linear(hid_size, dict_size)
        )

    def encode(self, x):
        # LSTM returns tuple: output, (h_cell, hidden_state)
        # Output, (hidden, cell) = LSTM(input, (hidden, cell) )
        _, hid = self.encoder(x)
        return hid

    def get_encoded_item(self, encoded, index):
        # Utility, to access hidden state of an individual component within the input batch
        # Extract the hidden state for the index'th element within the Batch
        return encoded[0][:, index:index+1].contiguous(), \
                encoded[1][:, index:index+1].contiguous()

    def decode_teacher(self, hid, input_seq):
        # Method assumes batch size=1
        out, _ = self.decoder(input_seq)
        # Return Logit
        out = self.output(out.data)
        return out

    def decode_one(self, hid, input_x):
        out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
        out = self.output(out)
        return out.squeeze(dim=0), new_hid

    def decode_chain_argmax(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the reccuring network.
        Act greedily i.e. arg_max: Pr(word)
        """
        res_logits = []
        res_tokens = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_token_v = torch.max(out_logits, dim=1)[1]
            out_token = out_token_v.data.cpu().numpy()[0]

            cur_emb = self.emb(out_token_v)

            res_logits.append(out_logits)
            res_tokens.append(out_token)
            if stop_at_token is not None and out_token == stop_at_token:
                break

        # Concatonate sequence
        return torch.cat(res_logits), res_tokens








