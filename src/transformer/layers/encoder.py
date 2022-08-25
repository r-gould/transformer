import torch
import torch.nn as nn

from .sublayers import (PositionalEncoder, MultiHeadAttention, 
                        AddNorm, FeedForward)

class Encoder(nn.Module):

    def __init__(self, num_layers, num_heads, d_model, d_v, d_k, d_ff, 
                dropout, vocab_in, max_seq_len, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_in, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoder(d_model, max_seq_len)
        self.drop = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(num_heads, d_model, d_v, d_k, d_ff, dropout) 
            for _ in range(num_layers)]
        )

    def forward(self, src, pad_mask):

        embeds = self.embedding(src)
        embeds = self.pos_encoder(embeds)
        embeds = self.drop(embeds)

        out = embeds
        for layer in self.encoder_layers:
            out = layer(out, pad_mask)

        return out

class EncoderLayer(nn.Module):

    def __init__(self, num_heads, d_model, d_v, d_k, d_ff, dropout):

        super().__init__()

        self.attention = MultiHeadAttention(num_heads, d_model, d_v, d_k)
        self.add_norm_1 = AddNorm(d_model, dropout)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, encoder_in, pad_mask):
        
        attn = self.attention(encoder_in, encoder_in, encoder_in, pad_mask)
        attn = self.add_norm_1(attn, encoder_in)

        out = self.feed_forward(attn)
        return self.add_norm_2(out, attn)