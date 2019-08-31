import torch
from torch.utils.data.sampler import SubsetRandomSampler
import os
import csv
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from utils import compute_overlap
from torchvision import transforms
from PIL import Image

np.random.seed(42)


class NSGVDataset(torch.utils.data.Dataset):
    def __init__(self, textual_data_path: str, visual_data_path: str, num_time_scales: int, delta: int,
                 threshold: float):
        """
        :param textual_data_path: directory containing the annotations
        :param visual_data_path: directory containing the videos
        :param num_time_scales: K in the paper
        :param delta: ẟ in the paper
        :param threshold: θ in the paper
        """
        super(NSGVDataset, self).__init__()
        self.num_time_scales = num_time_scales
        self.delta = delta
        self.textual_data_path = textual_data_path
        self.visual_data_path = visual_data_path
        self.threshold = threshold
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        files = os.listdir(textual_data_path)
        if '.DS_Store' in files: files.remove('.DS_Store')

        self.textual_data = []
        tokenizer = ToktokTokenizer()

        for file in files:
            with open(os.path.join(textual_data_path, file)) as tsvfile:
                video_id = file.replace('.aligned.tsv', '')
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    start_frame, end_frame = int(row[0]), int(row[1])
                    sents = set([sent for sent in row[6:] if len(sent) > 0])
                    sents = [tokenizer.tokenize(sent.lower()) for sent in sents]
                    self.textual_data += [(video_id, start_frame, end_frame, sent) for sent in sents]

    def __len__(self):
        return len(self.textual_data)

    def _generate_label(self, T, start_frame, end_frame):
        """
        :param T: number of the frames of video
        :param start_frame: The corresponding start frame for the annotation
        :param end_frame:  The corresponding ending frame for the annotation
        :return: label with shape (T, K) where T is the length of the visual_input
        """
        label = torch.zeros([T, ], dtype=torch.int32)
        for t in range(T):
            for k in range(self.num_time_scales):
                if compute_overlap(t - k * self.delta, t, start_frame, end_frame) > self.threshold:
                    label[t, k] = 1

        return label

    def __getitem__(self, item):

        video_id, start_frame, end_frame = self.textual_data[item][0], self.textual_data[item][1], \
                                           self.textual_data[item][2]
        visual_data = np.load(os.path.join(self.visual_data_path, video_id + '.npy'))

        visual_data_tensor = torch.cat([self.transforms(img).unsqueeze(dim=0) for img in visual_data], dim=0)

        label = self._generate_label(visual_data_tensor.shape[0], start_frame, end_frame)

        return {'visual data': visual_data_tensor, 'textual data': self.textual_data[item], 'label': label}


if __name__ == '__main__':
    data = NSGVDataset(textual_data_path='data/textual_data/TACoS', visual_data_path='data/processed_visual_data/TACoS',
                       num_time_scales=10, delta=4, threshold=1.)
    #a = data[14329]
    #print(a['visual data'].shape)
    num_data = len(data)

    num_train, num_val = int(num_data * 0.9), int(num_data * 0.005)
    num_test = num_data - num_train - num_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [num_train, num_val, num_test])
    print(len(test_dataset))
    print(len(train_dataset))
    print(len(val_dataset))





