# TODOs
# CUDA - GPUs

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
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

    # TODO: For RL - Treat Sequential Decision Making of Selecting a Token as an MDP process (Policy)?
    def decode_chain_sampling(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted to the net again.
        Act probabalistically
        """
        res_logits = []
        res_actions = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            # Convert logits to Probabilities using Softmax
            out_probs_v = F.softmax(out_logits, dim=1)
            out_probs = out_probs_v.cpu().numpy[0]
            # Take Action: Randomly Sample according to probability determined weights (Stochastic MDP)
            action = int(np.random.choice(out_probs.shape[0], p=out_probs))
            action_v = torch.LongTensor([action]).to(begin_emb.device)

            cur_emb = self.emb(out_logits)

            res_logits.append(out_logits)
            res_actions.append(action)

            if stop_at_token is not None and action == stop_at_token:
                break

        return torch.cat(res_logits), res_actions

# TODO: Print procedure
# Note: batch == the input batch
# Note: the input batch is a list of (phrase, reply) tuples
# 1. We sort the batch by the first phrase length, in decreasing order - requirement of CuDNN Library
# 2. Create a matrix with [batch, max_input_phrase]
def pack_batch_no_out(batch, embeddings, device="cpu"):
    assert isinstance(batch, list)
    # Sort Descending
    batch.sort(key=lambda s: len(s[0], reverse=True))
    # Zip (phrase, reply) as (x, y) pairs - list of tuples
    input_idx, output_idx = zip(*batch)

    # Create Padded Matrix of inputs
    lens = list(map(len, input_idx))
    # Create Padded Sequence (max len Matrix)
    # Our sequences of variable length are padded with zeros to the longest sequence
    input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
    for idx, x in enumerate(input_idx):
        input_mat[idx, :len(x)] = x
    input_v = torch.tensor(input_mat).to(device)
    input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)

    # Look up Embeddings
    emb_input_seq = rnn_utils.PackedSequence(r, input_seq.batch_sizes)

    return emb_input_seq, input_idx, output_idx

def pack_input(input_data, embeddings, device="cpu"):
    input_v = torch.LongTensor([input_data]).to(device)
    r = embeddings(input_v)
    return rnn_utils.pack_padded_sequence(r, [len(input_data)])

def pack_batch(batch, embeddings, device="cpu"):
    emb_input_seq, input_idx, output_idx = pack_batch_no_out(batch, embeddings, device)

    # Prepare output sequences, with end token stripped
    output_seq_list = []
    for out in output_idx:
        output_seq_list.append(pack_input(out[:-1], embeddings, device))
    return emb_input_seq, output_seq_list, input_idx, output_idx

def seq_bleu(model_out, ref_seq):
    model_seq = torch.max(model_out.data, dim=1)
    # Convert to numpy array
    model_seq = model_seq.cpu().numpy()

    return utils.calc_bleu(model_seq, ref_seq)

















