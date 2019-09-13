import torch
import torch.nn as nn
from torch.nn import LSTM, LSTMCell
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import List


class TextualLSTMEncoder(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int):
        """
        :param embed_size: embedding dimension of the words
        :param hidden_size: hidden size of the LSTM encoder
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.encoder = LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=False)

    def forward(self, input: torch.Tensor, lengths: List[int]):
        """
        :param input: A batch of sentences with shape (n_batch, N, embed_size)
        :param lengths: lengths of the input sentences
        :return: the encoded representation of the sentences
        """
        x = input.permute(1, 0, 2)  # shape (N, n_batch, embed_size)
        x = pack_padded_sequence(x, lengths)
        enc_hiddens, (_, _) = self.encoder(x)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens)  # shape of enc_hiddens is (N, n_batch, hidden_size)

        return enc_hiddens.permute(1, 0, 2)  # shape (n_batch, N, hidden_size)


