import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder
from models.interactor import Interactor
from models.visual_lstm_encoder import VisualLSTMEncoder
from models.textual_lstm_encoder import TextualLSTMEncoder
from models.grounder import Grounder


class tgn(nn.Module):
    def __init__(self, args):
        super(tgn, self).__init__()
        self.textual_lstm_encoder = TextualLSTMEncoder(embed_size=, hidden_size=)
        self.cnn_encoder = CNNEncoder()
        self.visual_lstm_encoder = VisualLSTMEncoder(embed_size=)
        self.grounder = Grounder(input_size=None, K=)
        self.interactor = Interactor(hidden_size_ilstm=, hidden_size_visual=,
                                     hidden_size_textual=)

    def forward(self, visual_input: torch.Tensor, textual_input: torch.Tensor):
        """

        :param visual_input:
        :param textual_input:
        :return:
        """
        h_v = self.cnn_encoder(visual_input)
        h_s = self.textual_lstm_encoder(textual_input)
        h_r = self.interactor(h_v, h_s)
        scores = self.grounder(h_r)

        return scores





