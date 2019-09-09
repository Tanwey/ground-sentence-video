import torch
import torch.nn as nn
from models.cnn_encoder import VGG16, InceptionV4
from models.interactor import Interactor
from models.visual_lstm_encoder import VisualLSTMEncoder
from models.textual_lstm_encoder import TextualLSTMEncoder
from models.grounder import Grounder
from utils import pad_visual_data
from typing import List, Dict, Tuple
import numpy as np
import sys


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
        self.num_time_scales = int(args['--num-time-scales'])
        self.textual_lstm_encoder = TextualLSTMEncoder(embed_size=word_embed_size,
                                                       hidden_size=hidden_size_textual)
        self.cnn_encoder = VGG16()

        self.feature_size = self.cnn_encoder.model.classifier[-3].out_features

        self.visual_lstm_encoder = VisualLSTMEncoder(input_size=self.feature_size, hidden_size=hidden_size_visual)

        self.grounder = Grounder(input_size=hidden_size_ilstm,
                                 num_time_scales=self.num_time_scales)

        self.interactor = Interactor(hidden_size_ilstm=hidden_size_ilstm,
                                     hidden_size_visual=hidden_size_visual,
                                     hidden_size_textual=hidden_size_textual)

    def forward(self, visual_input, textual_input: torch.Tensor, lengths_t: List[int]):
        """
        :param visual_input: a tensor containing a batch of input images (n_batch, T, 224, 224, 3)
        :param textual_input: a tensor containing a batch of embedded words
        with shape (n_batch, N, sentence_length, word_embed_size: 300)
        :param lengths_t:
        :param y:
        :param videos_mask:
        :return: grounding scores with shape (n_batch, T, K)
        """
        lengths_v = [v.shape[0] for v in visual_input]
        mask = self._generate_videos_mask(lengths_v)

        visual_input_cat = torch.cat(visual_input, dim=0).to(torch.float32)

        features_v_cat = self.cnn_encoder(visual_input_cat)  # shape: (n_batch, T, feature_size)
        features_v = torch.split(features_v_cat, lengths_v)

        features_v = sorted(features_v, key=lambda v: v.shape[0], reverse=True)
        lengths_v = sorted(lengths_v, reverse=True)
        features_v_padded = pad_visual_data(features_v)  # shape (n_batch, T, dim_feature)

        h_s = self.textual_lstm_encoder(textual_input, lengths_t)  # shape: (n_batch, N, hidden_size_textual)

        h_v = self.visual_lstm_encoder(features_v_padded, lengths_v)  # shape: (n_batch, T, hidden_size_visual)

        h_r = self.interactor(h_v, h_s)  # shape: (n_batch, T, hidden_size_ilstm)

        probs = self.grounder(h_r)

        return probs, mask

    def _generate_videos_mask(self, lengths: List[int]):
        n_batch = len(lengths)
        max_len = np.max(lengths)

        mask = torch.ones(n_batch, max_len, self.num_time_scales)

        for i in range(len(lengths)):
            mask[i, lengths[i]:, :] = 0

        return mask.to(self.device)


    @property
    def device(self) -> torch.device:
        """
        Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.grounder.projection.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = TGN()
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
