import torch
import torch.nn as nn
from torchvision.models import vgg16


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = vgg16(pretrained=True, progress=True)
        features = list(self.model.classifier.children())[:-1]  # removing the last layer
        self.model.classifier = nn.Sequential(*features)

        print(self.model.classifier[0])

        # freeze the weights of the network so it will not be trained
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor):
        """
        :param input: A batch of images with shape (K, 224, 224, 3)
        :return: extracted features with shape (K, 4096)
        """
        self.model.eval()
        return torch.zeros([input.shape[0], 4096])
        # return self.model(input)


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



if __name__ == '__main__':
    model = VGG16()