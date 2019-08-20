import torch
import torch.nn as nn
from torch.nn import LSTM, LSTMCell



class TextualLSTMEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        """
        :param embed_size: embedding dimension of the words
        :param hidden_size: hidden size of the LSTM encoder
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.encoder = LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=False)

    def forward(self, input: torch.Tensor):
        """
        :param input: A batch of sentences with shape
        :return: the encoded representation of the sentences
        """
        outputs, (_, _) = self.encoder(input)

        return outputs


