import torch
import torch.nn as nn
from torch.nn import Linear


class Grounder(nn.Module):
    def __init__(self, input_size: int, K: int):
        super(Grounder, self).__init__()
        self.projection = Linear(input_size, K, bias=True)

    def forward(self, input: torch.Tensor):
        """
        :param input: Output of iLSTM with the shape (batch_size, T, hidden_size_iLSTM)
        :return C_T: torch.Tensor with shape (n_batch, T, num_time_scales)
        """
        C_t = torch.sigmoid(self.projection(input))

        return C_t
