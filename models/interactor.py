import torch
import torch.nn as nn
from torch.nn import LSTM, LSTMCell, Linear, Parameter


class Interactor(nn.Module):
    def __init__(self, hidden_size_textual: int, hidden_size_visual: int,
                 hidden_size_ilstm: int):
        """
        :param input_size:
        :param hidden_size:
        """
        super(Interactor, self).__init__()

        # represented by W_S, W_R, W_V with bias b
        self.projection_S = Linear(hidden_size_textual, hidden_size_ilstm, bias=True)
        self.projection_V = Linear(hidden_size_visual, hidden_size_ilstm, bias=True)
        self.projection_R = Linear(hidden_size_ilstm, hidden_size_ilstm, bias=True)

        # parameter w with bias c
        self.projection_w = Linear(hidden_size_ilstm, 1, bias=True)

        self.hidden_size_textual = hidden_size_textual
        self.hidden_size_visual = hidden_size_visual
        self.hidden_size_ilstm = hidden_size_ilstm

        self.iLSTM = LSTMCell(input_size=hidden_size_textual+hidden_size_visual,
                              hidden_size=hidden_size_ilstm)

    def forward(self, h_v: torch.Tensor, h_s: torch.Tensor):
        """
        :param h_v: with shape (n_batch, T, hidden_size_visual)
        :param h_s: with shape (n_batch, N, hidden_size_textual)
        :return: outputs of the iLSTM with shape (n_batch, T, hidden_size_ilstm)
        """
        n_batch, T, N = h_v.shape[0], h_v.shape[1], h_s.shape[1]

        # h_r_{t-1} in the paper
        h_r_prev = torch.zeros([n_batch, self.hidden_size_ilstm], device=self.device)
        c_r_prev = torch.zeros([n_batch, self.hidden_size_ilstm], device=self.device)

        outputs = []

        for t in range(T):
            beta_t = self.projection_w(torch.tanh(self.projection_R(h_r_prev).unsqueeze(dim=1) +
                                                  self.projection_S(h_s) +
                                                  self.projection_V(h_v[:, t, :]).unsqueeze(dim=1))
                                       ).squeeze(2)  # shape (n_batch, N)

            #print('beta_t shape', beta_t.shape)

            alpha_t = torch.softmax(beta_t, dim=0)  # shape: (n_batch, N)

            # H_ts_s with shape (n_batch, hidden_size_textual)
            H_t_s = torch.bmm(h_s.permute(0, 2, 1), alpha_t.unsqueeze(dim=2)).squeeze(dim=2)
            #print('H_t_s shape', H_t_s.shape)

            r_t = torch.cat([h_v[:, t, :], H_t_s], dim=1)  # shape (n_batch, hidden_size_textual+hidden_size_visual)
            #print('r_t shape', r_t.shape)

            h_r_new, c_r_new = self.iLSTM(r_t, (h_r_prev, c_r_prev))
            outputs.append(h_r_new.unsqueeze(1))
            h_r_prev, c_r_prev = h_r_new, c_r_new

        return torch.cat(outputs, dim=1)

    @property
    def device(self) -> torch.device:
        """
        Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.projection_S.weight.device
