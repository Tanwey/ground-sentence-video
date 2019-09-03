import torch
from torch.utils.data.sampler import SubsetRandomSampler
import os
import csv
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from utils import compute_overlap
from torchvision import transforms
from math import ceil
from typing import List
from collections import namedtuple

np.random.seed(42)


Annotation = namedtuple('Annotation', ['video_id', 'start_frame', 'end_frame', 'sent'])


class NSGVDataset(torch.utils.data.Dataset):
    def __init__(self, textual_data_path: str, visual_data_path: str, num_time_scales: int, delta: int,
                 threshold: float, val_ratio=0.25, test_ratio=0.25):
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

        index_array = list(range(len(self.textual_data)))
        np.random.shuffle(index_array)

        val_size = val_ratio * len(index_array)
        test_size = test_ratio * len(index_array)
        self.val_indices = index_array[:val_size],
        self.test_indices = index_array[val_size:val_size+test_size]
        self.train_indices = index_array[val_size+test_size:]

        # loading all of the preprocessed videos as numpy nd-arrays
        self.visual_data = dict()
        files = os.listdir(textual_data_path)
        if '.DS_Store' in files: files.remove('.DS_Store')

        for file in files:
            path = os.path.join(visual_data_path, file)
            self.visual_data[file] = np.load(path)

    def __len__(self):
        return len(self.textual_data)

    def _generate_labels(self, visual_data: List[np.ndarray], textual_data):
        """
        :param T: number of the frames of video
        :param start_frame: The corresponding start frame for the annotation
        :param end_frame:  The corresponding ending frame for the annotation
        :return: label with shape (T, K) where T is the length of the visual_input
        """
        labels = []
        for s, v in zip(visual_data, textual_data):
            T = v.shape[0]
            label = torch.zeros([T, self.num_time_scales], dtype=torch.int32)
            for t in range(T):
                start_frame, end_frame = s[1], s[2]
                for k in range(self.num_time_scales):
                    if compute_overlap(t - k * self.delta, t, start_frame, end_frame) > self.threshold:
                        label[t, k] = 1
            labels.append(label)

        return labels

    def __getitem__(self, item):

        video_id, start_frame, end_frame = self.textual_data[item][0], self.textual_data[item][1], \
                                           self.textual_data[item][2]
        visual_data = np.load(os.path.join(self.visual_data_path, video_id + '.npy'))
        visual_data_tensor = torch.cat([self.transforms(img).unsqueeze(dim=0) for img in visual_data], dim=0)
        label = self._generate_label(visual_data_tensor.shape[0], start_frame, end_frame)

        return {'visual data': visual_data_tensor, 'textual data': self.textual_data[item], 'label': label}

    def data_iter(self, batch_size: int, set: str):

        index_array = None

        if set == 'train':
            index_array = self.train_indices
        elif set == 'val':
            index_array = self.val_indices

        batch_num = ceil(len(index_array) / batch_size)

        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            textual_data = [self.textual_data[idx] for idx in indices]
            textual_data = sorted(textual_data, key=lambda e: len(e[3]), reverse=True)
            visual_data = [self.visual_data[sent[0]] for sent in textual_data]
            labels = self._generate_labels(visual_data=visual_data, textual_data=textual_data)
            yield textual_data, visual_data, labels


if __name__ == '__main__':
    data = NSGVDataset(textual_data_path='data/textual_data/TACoS', visual_data_path='data/processed_visual_data/TACoS',
                       num_time_scales=10, delta=4, threshold=1.)
    print(len(data))
    #a = data[14329]







