import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder
from models.interactor import Interactor
from models.visual_lstm_encoder import VisualLSTMEncoder
from models.textual_lstm_encoder import TextualLSTMEncoder
from models.grounder import Grounder


class TGN(nn.Module):
    def __init__(self, args):
        """
        :param args:
        """
        super(TGN, self).__init__()

        word_embed_size = args['word-embed-size']
        hidden_size_textual = args['hidden-size-textual']
        hidden_size_visual = args['hidden-size-visual']
        hidden_size_ilstm = args['hidden-size-ilstm']

        self.textual_lstm_encoder = TextualLSTMEncoder(embed_size=word_embed_size,
                                                       hidden_size=hidden_size_textual)
        self.cnn_encoder = CNNEncoder()

        self.visual_lstm_encoder = VisualLSTMEncoder(embed_size=hidden_size_visual)

        self.grounder = Grounder(input_size=hidden_size_ilstm,
                                 num_time_scales=args['num-time-scales'])

        self.interactor = Interactor(hidden_size_ilstm=hidden_size_ilstm,
                                     hidden_size_visual=hidden_size_visual,
                                     hidden_size_textual=hidden_size_textual)

    def forward(self, visual_input: torch.Tensor, textual_input: torch.Tensor):
        """

        :param visual_input: a tensor containing a batch of input images (batch, )  # TODO: specify the shape
        :param textual_input: a tensor containing a batch of words in the format of their
        embeddings with shape (batch, N, word_embed_size)
        :return:
        """
        features_v = self.cnn_encoder(visual_input)
        h_s = self.textual_lstm_encoder(textual_input)
        h_v = self.visual_lstm_encoder(features_v)
        h_r = self.interactor(h_v, h_s)
        scores = self.grounder(h_r)

        return scores





