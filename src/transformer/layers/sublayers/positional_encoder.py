import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len):
        
        super().__init__()

        self.encoding = self.init_encoding(d_model, max_seq_len)

    def forward(self, embeds):

        _, seq_len, _ = embeds.shape
        return embeds + self.encoding[:seq_len].to(embeds.device)

    @staticmethod
    def init_encoding(d_model, max_seq_len):

        encoding = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).reshape(-1, 1)
        idx = torch.arange(0, d_model)

        encoding[:, 0::2] = torch.sin(pos / 10000**(idx[0::2] / d_model))
        encoding[:, 1::2] = torch.cos(pos / 10000**((idx[1::2]-1) / d_model))
        return encoding