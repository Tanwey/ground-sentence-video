import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder
from models.interactor import InteractionLSTM
from models.visual_lstm_encoder import VisualLSTMEncoder
from models.textual_lstm_encoder import TextualLSTMEncoder


class tgn(nn.Module):
    def __init__(self):
        super(tgn, self).__init__()
        self.textual_lstm_encoder = TextualLSTMEncoder(embed_size=, hidden_size=)
        self.cnn_encoder = CNNEncoder()
        self.visual_lstm_encoder = VisualLSTMEncoder()

    def forward(self, h_video: torch.Tensor, h_sentence: torch.Tensor):
        T, N = h_video.shape[0], h_sentence.shape[0]

        for t in range(T):
            beta_t =
