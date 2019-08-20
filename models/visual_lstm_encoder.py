import torch
import torch.nn as nn
from torch.nn import LSTM


class visualLSTMEncoder(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.encoder = LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=False)

    def forward(self, input: torch.Tensor):
        """
        :param input: A batch of frame features with shape ()
        :return: the encoded representation of the frames
        """
        outputs, (_, _) = self.encoder(input)

        return outputs