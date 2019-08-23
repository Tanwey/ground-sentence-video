import torch
import torch.nn as nn
from torch.nn import LSTM


class VisualLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.encoder = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=False)

    def forward(self, input: torch.Tensor):
        """
        :param input: A batch of frame features with shape ()
        :return: the encoded representation of the frames
        """
        outputs, (_, _) = self.encoder(input)

        return outputs