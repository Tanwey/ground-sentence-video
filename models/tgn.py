import torch
import torch.nn as nn
from models.cnn_encoder import VGG16, InceptionV4
from models.interactor import Interactor
from models.visual_lstm_encoder import VisualLSTMEncoder
from models.textual_lstm_encoder import TextualLSTMEncoder
from models.grounder import Grounder
from utils import pad_visual_data
from typing import List, Dict


class TGN(nn.Module):
    def __init__(self, args: Dict):
        """
        :param args: Dict of experimental settings
        """
        super(TGN, self).__init__()

        word_embed_size = int(args['--word-embed-size'])
        hidden_size_textual = int(args['--hidden-size-textual-lstm'])
        hidden_size_visual = int(args['--hidden-size-visual-lstm'])
        hidden_size_ilstm = int(args['--hidden-size-ilstm'])

        self.textual_lstm_encoder = TextualLSTMEncoder(embed_size=word_embed_size,
                                                       hidden_size=hidden_size_textual)
        self.cnn_encoder = VGG16()

        self.feature_size = self.cnn_encoder.model.classifier[-3].out_features

        self.visual_lstm_encoder = VisualLSTMEncoder(input_size=self.feature_size, hidden_size=hidden_size_visual)

        self.grounder = Grounder(input_size=hidden_size_ilstm,
                                 num_time_scales=int(args['--num-time-scales']))

        self.interactor = Interactor(hidden_size_ilstm=hidden_size_ilstm,
                                     hidden_size_visual=hidden_size_visual,
                                     hidden_size_textual=hidden_size_textual)

    def forward(self, visual_input, textual_input: torch.Tensor, lengths_t: List[int]):
        """
        :param visual_input: a tensor containing a batch of input images (n_batch, T, 224, 224, 3)
        :param textual_input: a tensor containing a batch of embedded words
        with shape (n_batch, N, sentence_length, word_embed_size: 300)
        :param lengths_t:
        :return: grounding scores with shape (n_batch, T, K)
        """
        lengths_v = [v.shape[0] for v in visual_input]
        visual_input_cat = torch.cat(visual_input, dim=0).permute(0, 3, 1, 2).to(torch.float32)

        features_v_cat = self.cnn_encoder(visual_input_cat)  # shape: (n_batch, T, feature_size)
        print('shape of cat features', features_v_cat.shape)
        features_v = torch.split(features_v_cat, lengths_v)

        features_v = sorted(features_v, key=lambda v: v.shape[0], reverse=True)
        lengths_v = sorted(lengths_v, reverse=True)
        features_v_padded = pad_visual_data(features_v)  # shape (n_batch, T, dim_feature)

        print('features_v padded shape', features_v_padded.shape)

        h_s = self.textual_lstm_encoder(textual_input, lengths_t)  # shape: (n_batch, N, hidden_size_textual)
        print('h_s shape', h_s.shape)

        h_v = self.visual_lstm_encoder(features_v_padded, lengths_v)  # shape: (n_batch, T, hidden_size_visual)
        print('h_v shape', h_v.shape)

        h_r = self.interactor(h_v, h_s)  # shape: (n_batch, T, hidden_size_ilstm)
        print('h_r shape', h_r.shape)

        scores = self.grounder(h_r)
        print('scores shape', scores.shape)

        return scores
