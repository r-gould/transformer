import torch
import torch.nn as nn

from .sublayers import (PositionalEncoder, MultiHeadAttention, 
                        AddNorm, FeedForward)

class Decoder(nn.Module):

    def __init__(self, num_layers, num_heads, d_model, d_v, d_k, d_ff, 
                dropout, vocab_out, max_seq_len, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_out, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoder(d_model, max_seq_len)
        self.drop = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(num_heads, d_model, d_v, d_k, d_ff, dropout)
            for _ in range(num_layers)]
        )

    def forward(self, trg, encoder_out, src_pad_mask, trg_pad_mask):

        embeds = self.embedding(trg)
        embeds = self.pos_encoder(embeds)
        embeds = self.drop(embeds)

        out = embeds
        for layer in self.decoder_layers:
            out = layer(out, encoder_out, src_pad_mask, trg_pad_mask)

        return out

class DecoderLayer(nn.Module):

    def __init__(self, num_heads, d_model, d_v, d_k, d_ff, dropout):
        
        super().__init__()

        self.masked_attention = MultiHeadAttention(num_heads, d_model, d_v, d_k, masked=True)
        self.add_norm_1 = AddNorm(d_model, dropout)

        self.attention = MultiHeadAttention(num_heads, d_model, d_v, d_k)
        self.add_norm_2 = AddNorm(d_model, dropout)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.add_norm_3 = AddNorm(d_model, dropout)

    def forward(self, decoder_in, encoder_out, src_pad_mask, trg_pad_mask):

        masked_attn = self.masked_attention(decoder_in, decoder_in, decoder_in, trg_pad_mask)
        masked_attn = self.add_norm_1(masked_attn, decoder_in)

        attn = self.attention(masked_attn, encoder_out, encoder_out, src_pad_mask)
        attn = self.add_norm_2(attn, masked_attn)

        out = self.feed_forward(attn)
        return self.add_norm_3(out, attn)