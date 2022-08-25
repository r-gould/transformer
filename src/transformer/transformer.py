import torch
import torch.nn as nn

from .layers import Encoder, Decoder

class Transformer(nn.Module):

    def __init__(self, enc_layers, dec_layers, num_heads, d_model, d_v, d_k, d_ff, 
                dropout, vocab_in, vocab_out, max_seq_len, pad_idx):

        super().__init__()

        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        self.encoder = Encoder(enc_layers, num_heads, d_model, d_v, d_k, d_ff,
                                dropout, vocab_in, max_seq_len, pad_idx)

        self.decoder = Decoder(dec_layers, num_heads, d_model, d_v, d_k, d_ff,
                                dropout, vocab_out, max_seq_len, pad_idx)

        self.output = nn.Linear(d_model, vocab_out)

    def forward(self, src, trg):

        src_pad_mask = (src == self.pad_idx)
        trg_pad_mask = (trg == self.pad_idx)

        encoder_out = self.encoder(src, src_pad_mask)
        decoded = self.decoder(trg, encoder_out, src_pad_mask, trg_pad_mask)
        return self.output(decoded)