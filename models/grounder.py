import torch
import torch.nn as nn
from torch.nn import Linear


class Grounder(nn.Module):
    def __init__(self, input_size: int, num_time_scales: int):
        super(Grounder, self).__init__()
        self.projection = Linear(input_size, num_time_scales, bias=True)

    def forward(self, input: torch.Tensor):
        """
        :param input: Output of iLSTM with the shape (batch_size, T, hidden_size_iLSTM)
        :return:
        """
        C_t = torch.sigmoid(self.projection(input))

        return C_t
