import torch
import torch.nn as nn
from torch.nn import LSTM, LSTMCell, Linear, Parameter

class InteractionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size_textual: int, hidden_size_visual: int,
                 hidden_size_ilstm: int):
        """
        :param input_size:
        :param hidden_size:
        """
        super(InteractionLSTM, self).__init__()
        self.projection_S = Linear(hidden_size_textual, hidden_size_ilstm)
        self.projection_V = Linear(hidden_size_visual, hidden_size_ilstm)
        self.projection_R = Linear(hidden_size_ilstm, hidden_size_ilstm)

        self.w = Parameter(requires_grad=True)
        self.b = Parameter(requires_grad=True)
        self.c = Parameter(requires_grad=True)

        self.hidden_size_textual = hidden_size_textual
        self.hidden_size_visual = hidden_size_visual
        self.hidden_size_ilstm = hidden_size_ilstm

        self.iLSTM = LSTMCell(input_size=hidden_size_textual+hidden_size_visual,
                              hidden_size=hidden_size_ilstm)

    def forward(self, h_v: torch.Tensor, h_s: torch.Tensor):
        """
        :param h_v: with shape (batch_size, hidden_size_visual, T)
        :param h_s: with shape (batch_size, hidden_size_textual, N)
        :return:
        """
        batch_size, T, N = h_v.shape[0], h_v.shape[1], h_s.shape[1]
        h_r_prev = torch.zeros([batch_size, 1, self.hidden_size_ilstm])

        for t in range(T):
            beta_t =  torch.dot(self.w, torch.tanh(self.projection_R(h_r_prev) +
                                                   self.projection_S(h_s[t])))
