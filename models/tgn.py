import torch
import torch.nn as nn
from models.cnn_encoder import VGG16, InceptionV4
from models.interactor import Interactor
from models.visual_lstm_encoder import VisualLSTMEncoder
from models.textual_lstm_encoder import TextualLSTMEncoder
from models.grounder import Grounder
from typing import List, Dict, Tuple
import numpy as np
import sys


class TGN(nn.Module):
    def __init__(self, word_embed_size: int, hidden_size_textual: int, hidden_size_visual: int,
                 hidden_size_ilstm: int, K: int, visual_feature_size: int):
        super(TGN, self).__init__()

        self.word_embed_size = word_embed_size
        self.hidden_size_visual = hidden_size_visual
        self.hidden_size_textual = hidden_size_textual
        self.hidden_size_ilstm = hidden_size_ilstm
        self.K = K

        self.textual_lstm_encoder = TextualLSTMEncoder(embed_size=word_embed_size,
                                                       hidden_size=hidden_size_textual)

        self.visual_feature_size = visual_feature_size

        self.visual_lstm_encoder = VisualLSTMEncoder(input_size=self.visual_feature_size, hidden_size=hidden_size_visual)

        self.grounder = Grounder(input_size=hidden_size_ilstm, K=K)

        self.interactor = Interactor(hidden_size_ilstm=hidden_size_ilstm,
                                     hidden_size_visual=hidden_size_visual,
                                     hidden_size_textual=hidden_size_textual)

    def forward(self, features_v: List[torch.Tensor], textual_input: torch.Tensor, lengths_t: List[int]):
        """
        :param features_v: visual features extracted by a CNN encoder beforehand
        :param textual_input: a tensor containing a batch of embedded words
        with shape (n_batch, N, max_sentence_length, word_embed_size)
        :param lengths_t: lengths of sentences
        :returns grounding scores as a tensor with shape (n_batch, T, K)
        """
        lengths_v = [v.shape[0] for v in features_v]
        mask = self._generate_visual_mask(lengths_v)  # used later for computing the loss

        features_v = sorted(features_v, key=lambda v: v.shape[0], reverse=True)
        lengths_v = sorted(lengths_v, reverse=True)
        features_v_padded = self._pad_visual_data(features_v)  # shape (n_batch, T, dim_feature)

        h_s = self.textual_lstm_encoder(textual_input, lengths_t)  # shape: (n_batch, N, hidden_size_textual)

        h_v = self.visual_lstm_encoder(features_v_padded, lengths_v)  # shape: (n_batch, T, hidden_size_visual)

        h_r = self.interactor(h_v, h_s)  # shape: (n_batch, T, hidden_size_ilstm)

        probs = self.grounder(h_r)

        return probs, mask

    def _generate_visual_mask(self, lengths: List[int]):
        """Generate a mask to not consider the padding positions in videos while computing the final loss"""
        n_batch = len(lengths)
        max_len = np.max(lengths)

        mask = torch.ones(n_batch, max_len, self.K)

        for i in range(len(lengths)):
            mask[i, lengths[i]:, :] = 0

        return mask.to(self.device)

    def _pad_visual_data(self, visual_data: List[torch.Tensor]):
        """
        :param visual_data: list of visual features as torch.Tensor
        :returns: a tensor with shape (n_batch, max_len, feature_dim) where max_len is the
        maximum length of input videos
        """
        feature_dim = visual_data[0].shape[1]
        max_len = np.max([v.shape[0] for v in visual_data])

        visual_data_padded = list(map(lambda v: torch.cat([v.to(self.device),
                                                           torch.zeros([max_len - v.shape[0], feature_dim]).to(self.device)]
                                                          ).unsqueeze(dim=0), visual_data))

        return torch.cat(visual_data_padded, dim=0)  # tensor with shape (n_batch, max_len, feature_dim)

    @property
    def device(self) -> torch.device:
        """
        Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.grounder.projection.weight.device

    @staticmethod
    def load(model_path: str):
        """
        Load the model from a file.
        :param model_path: path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = TGN(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """
        Save the model to a file.
        :param path: path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(word_embed_size=self.word_embed_size, hidden_size_textual=self.hidden_size_visual,
                         hidden_size_visual=self.hidden_size_visual, hidden_size_ilstm=self.hidden_size_ilstm,
                         K=self.K, visual_feature_size=self.visual_feature_size),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
