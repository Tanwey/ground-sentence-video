import torch.nn as nn
from torchvision.models import vgg16


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = vgg16(pretrained=True, progress=True)
        features = list(self.model.classifier.children())[:-1]  # removing the last layer
        self.model.classifier = nn.Sequential(*features)

    def forward(self, input):
        """
        :param input: A batch of images with shape
        :return: extracted features from
        """
        self.model.eval()
        return self.model(input)


class InceptionV4(nn.Module):
    def __init__(self):
        super(InceptionV4, self).__init__()

    def forward(self, input):
        pass


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()

    def forward(self, input):
        pass
