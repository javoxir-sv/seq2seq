import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



# RNN Encoder
# 1-layer LSTM model to encode the English name.
class RNNEncoder(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, hidden_size:int, num_layers:int, pe=None):
        super().__init__()
        self.pe = pe
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
                input_size=embedding_dim, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        if self.pe is not None:
            x = self.pe(x)
        output, (hidden, cell) = self.lstm(x)
        logits = self.fc(output)
        return logits, (output, (hidden, cell))
            


# RNN Decoder
# Implement RNN decoder to produce translation.
class RNNDecoder(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, hidden_size:int, num_layers:int, pe=None):
        super().__init__()
        self.pe = pe
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
                )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        x = self.embedding(x)
        if self.pe is not None:
            x = self.pe(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, (hidden, cell)


class Seq2SeqTranslator(nn.Module):
    def __init__(self, encoder: RNNEncoder, decoder: RNNDecoder, train_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_encoder = train_encoder

        if not train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, src_x, tgt_y):
        if not self.train_encoder:
            with torch.no_grad():
                _, (_, (hidden, cell)) = self.encoder(src_x)
        else:
            _, (_, (hidden, cell)) = self.encoder(src_x)

        logits, _ = self.decoder(tgt_y, hidden, cell)
        return logits


        
# Attention Machine Translation
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, encoder_outputs, hidden):
        # encoder_outputs.shape -> torch.Size([batch_size, seq_len, hidden_size])
        # hidden -> torch.Size([batch_size, hidden_size])
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # torch.Size([batch_size, seq_len, hidden_size])
        alignment_score = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        # torch.Size([batch_size, seq_len, hidden_size*2]) -> torch.Size([batch_size, seq_len, hidden_size]) 
        alignment_score = alignment_score.permute(0, 2, 1)
        # torch.Size([batch_size, hidden_size, seq_len])
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # torch.Size([batch_size, 1, hidden_size])
        attention_weights = F.softmax(torch.bmm(v, alignment_score).squeeze(1), dim=1) 
        # torch.Size([batch_size, 1, seq_len]) -> torch.Size([batch_size, seq_len])
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # torch.Size([batch_size, 1, seq_len]) @ torch.Size([batch_size, seq_len, hidden_size]) -> torch.Size([batch_size, 1, hidden_size])
        return context




class AttentionDecoder(nn.Module):
    def __init__(self, attention: Attention, vocab_size:int, embedding_dim:int, hidden_size:int, num_layers:int, pe=None):
        super().__init__()
        self.pe = pe
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
                input_size=embedding_dim + hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
                )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        logits = []
        embeds = self.embedding(x)
        if self.pe is not None:
            embeds = self.pe(embeds)

        for t in range(x.size(1)):
            embed = embeds[:, t].unsqueeze(1)
            prev_hidden = hidden[-1]
            context = self.attention(encoder_outputs, prev_hidden)
            lstm_input = torch.cat([embed, context], dim=-1)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            logit = self.fc(output)
            logits.append(logit)
        logits = torch.cat(logits, dim=-2)
        return logits, (hidden, cell)



class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder: RNNEncoder, decoder: AttentionDecoder, train_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_encoder = train_encoder

        if not train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, src_x, tgt_y):
        if not self.train_encoder:
            with torch.no_grad():
                _, (encoder_outputs, (hidden, cell)) = self.encoder(src_x)
        else:
            _, (encoder_outputs, (hidden, cell)) = self.encoder(src_x)
        logits, _ = self.decoder(tgt_y, hidden, cell, encoder_outputs)
        return logits


# Multi-head Attention Machine Translation
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads=3):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attention_list = []

        self.d_k = self.d_v = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(self.n_heads * self.d_k, d_model)

    def attention(self, query, key, value, mask):
        d_k = query.shape[-1]
        # (batch_size, head, seq_len, d_k)
        attention_values = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_values = attention_values.masked_fill(mask==0, -1e9)
        attention_values = attention_values.softmax(dim=-1)
        return (attention_values @ value), attention_values

    def forward(self, q, k, v, mask=None, reset_list=False):
        if reset_list:
            self.attention_list = []
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(2, 1)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(2, 1)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(2, 1)

        x, attention_values = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)
        self.attention_list.append(attention_values.detach().cpu())
        return self.w_o(x)


class MultiHeadAttnDecoder(nn.Module):
    def __init__(self, attention: MultiHeadAttention, vocab_size:int, embedding_dim:int, hidden_size:int, num_layers:int, pe=None):
        super().__init__()
        self.pe = pe
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
                input_size=embedding_dim + hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
                )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        logits = []
        embeds = self.embedding(x)
        if self.pe is not None:
            embeds = self.pe(embeds)
        key = encoder_outputs
        value = encoder_outputs
        mask = (encoder_outputs != 0).any(dim=-1).unsqueeze(1).unsqueeze(2)

        for t in range(x.size(1)):
            embed = embeds[:, t].unsqueeze(1)
            query = hidden[-1].unsqueeze(1)

            context = self.attention(query, key, value, mask, reset_list=(t==0))

            lstm_input = torch.cat([embed, context], dim=-1)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            logit = self.fc(output)
            logits.append(logit)
        logits = torch.cat(logits, dim=-2)
        return logits, (hidden, cell)


