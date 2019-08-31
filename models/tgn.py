import torch
import torch.nn as nn
from models.cnn_encoder import VGG16, InceptionV4
from models.interactor import Interactor
from models.visual_lstm_encoder import VisualLSTMEncoder
from models.textual_lstm_encoder import TextualLSTMEncoder
from models.grounder import Grounder

from typing import Dict


class TGN(nn.Module):
    def __init__(self, args: Dict):
        """
        :param args: Dict of experimental settings
        """
        super(TGN, self).__init__()

        word_embed_size = args['word-embed-size']
        hidden_size_textual = args['hidden-size-textual-lstm']
        hidden_size_visual = args['hidden-size-visual-lstm']
        hidden_size_ilstm = args['hidden-size-ilstm']

        self.textual_lstm_encoder = TextualLSTMEncoder(embed_size=word_embed_size,
                                                       hidden_size=hidden_size_textual)
        self.cnn_encoder = VGG16()

        self.feature_size = self.cnn_encoder.model.classifier[-1].out_features

        self.visual_lstm_encoder = VisualLSTMEncoder(input_size=self.feature_size, hidden_size=hidden_size_visual)

        self.grounder = Grounder(input_size=hidden_size_ilstm,
                                 num_time_scales=args['num-time-scales'])

        self.interactor = Interactor(hidden_size_ilstm=hidden_size_ilstm,
                                     hidden_size_visual=hidden_size_visual,
                                     hidden_size_textual=hidden_size_textual)

    def forward(self, visual_input: torch.Tensor, textual_input: torch.Tensor):
        """
        :param visual_input: a tensor containing a batch of input images (n_batch, T, 224, 224, 3)
        :param textual_input: a tensor containing a batch of embedded words
        with shape (n_batch, N, sentence_length, word_embed_size: 300)
        :return: grounding scores with shape (n_batch, T, K)
        """
        features_v = self.cnn_encoder(visual_input)  # shape: (n_batch, T, feature_size)
        h_s = self.textual_lstm_encoder(textual_input)  # shape: (n_batch, N, hidden_size_textual)
        h_v = self.visual_lstm_encoder(features_v)  # shape: (n_batch, T, hidden_size_visual)
        h_r = self.interactor(h_v, h_s)  # shape: (n_batch, T, hidden_size_ilstm)
        scores = self.grounder(h_r)

        return scores
