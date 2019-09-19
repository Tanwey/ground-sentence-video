import torch
import torch.nn as nn
from torch.nn import LSTM
from typing import List
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class VisualLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.encoder = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=False)

    def forward(self, x: torch.Tensor, lengths: List[int]):
        """
        :param input: A batch of frame features with shape (n_batch, T, feature_size)
        :param lengths:
        :return: the encoded representation of the frames
        """
        x = x.permute(1, 0, 2)  # shape (T, n_batch, feature_size)
        x = pack(x, lengths)
        enc_hiddens, (_, _) = self.encoder(x)
        enc_hiddens, _ = unpack(enc_hiddens)  # shape of enc_hiddens is (T, n_batch, hidden_size)

        return enc_hiddens.permute(1, 0, 2)  # shape (n_batch, T, hidden_size)
