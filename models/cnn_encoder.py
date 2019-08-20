import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg16


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        pass


    def forward(self, input):
        """
        :param input: A batch of images with shape
        :return: extracted features from
        """


