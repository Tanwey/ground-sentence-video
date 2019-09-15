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
from utils import pad_labels
import sys

np.random.seed(42)


Annotation = namedtuple('Annotation', ['video_id', 'start_frame', 'end_frame', 'sent'])


class TACoS(torch.utils.data.Dataset):
    def __init__(self, textual_data_path: str, visual_data_path: str, num_time_scales: int, delta: int,
                 threshold: float, val_ratio=0.25, test_ratio=0.25):
        """
        :param textual_data_path: directory containing the annotations
        :param visual_data_path: directory containing the processed videos
        :param num_time_scales: K in the paper
        :param delta: ẟ in the paper
        :param threshold: θ in the paper
        """
        super(TACoS, self).__init__()
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

        # loading the textual data
        print('Loading the textual data...', file=sys.stderr)
        for file in files:
            with open(os.path.join(textual_data_path, file)) as tsvfile:
                video_id = file.replace('.aligned.tsv', '')
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    start_frame, end_frame = int(row[0]), int(row[1])
                    sents = set([sent for sent in row[6:] if len(sent) > 0])
                    sents = [tokenizer.tokenize(sent.lower()) for sent in sents]
                    self.textual_data += [Annotation(video_id=video_id, start_frame=start_frame, end_frame=end_frame,
                                                     sent=sent) for sent in sents]

        index_array = list(range(len(self.textual_data)))
        np.random.shuffle(index_array)

        val_size = int(val_ratio * len(index_array))
        test_size = int(test_ratio * len(index_array))

        self.val_indices = index_array[:val_size]
        self.test_indices = index_array[val_size:val_size+test_size]
        self.train_indices = index_array[val_size+test_size:]

        print('val size is %d' % val_size, file=sys.stderr)
        print('test size is %d' % test_size, file=sys.stderr)
        print('train size is %d' % len(self.train_indices), file=sys.stderr)

        # loading all of the preprocessed videos as numpy nd-arrays
        # self.visual_data = dict()
        # files = os.listdir(visual_data_path)
        # if '.DS_Store' in files: files.remove('.DS_Store')

        # loading and preprocessing visual data
        #print('Loading the visual data...', file=sys.stderr)
        #for file in files:
        #    path = os.path.join(visual_data_path, file)
        #    video = np.load(path)
        #    self.visual_data[file.replace('.npy', '')] = self._transform(video)

    def __len__(self):
        return len(self.textual_data)

    def _generate_labels(self, visual_data: List[torch.Tensor], textual_data: List[Annotation]):
        """
        :param visual_data:
        :param textual_data:
        :return:
        """
        fps = 30
        sample_rate = fps * 5

        labels = []
        for v, s in zip(visual_data, textual_data):
            T = v.shape[0]
            label = torch.zeros([T, self.num_time_scales], dtype=torch.int32)
            start_frame, end_frame = s.start_frame, s.end_frame

            for t in range(T):
                for k in range(self.num_time_scales):
                    if (compute_overlap((t - (k+1) * self.delta) * sample_rate,
                                        t * sample_rate, start_frame, end_frame) / fps) > self.threshold:
                        label[t, k] = 1

            labels.append(label)

        return pad_labels(labels)  # torch.Tensor with shape (num_labels, T, K)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        s = self.textual_data[item]
        file = s.video_id + '.npy'
        path = os.path.join(self.visual_data_path, file)
        video = np.load(path)
        visual_data = self._transform(video)
        label = self._generate_labels([visual_data], [s])[0]
        return visual_data, s, label

    def _load_visual_data(self, textual_data: List[Annotation]):
        """
        :param textual_data:
        :return: list of the videos corresponding to the annotations
        """
        visual_data = []
        for t in textual_data:
            video_id = t.video_id
            file = video_id + '.npy'
            path = os.path.join(self.visual_data_path, file)
            video = np.load(path)
            visual_data.append(self._transform(video))

        return visual_data

    def data_iter(self, batch_size: int, set: str):
        """
        :param batch_size:
        :param set:
        :return:
        """
        index_array = None
        if set == 'train':
            index_array = self.train_indices
        elif set == 'val':
            index_array = self.val_indices

        batch_num = ceil(len(index_array) / batch_size)
        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            textual_data = [self.textual_data[idx] for idx in indices]
            textual_data = sorted(textual_data, key=lambda s: len(s.sent), reverse=True)

            visual_data = self._load_visual_data(textual_data)

            if set == 'train':
                labels = self._generate_labels(visual_data=visual_data, textual_data=textual_data)
                textual_data = [s.sent for s in textual_data]
                yield textual_data, visual_data, labels
            else:
                yield textual_data, visual_data

    def _transform(self, video):
        """
        :param video: np.ndarray with shape (T, 224, 224, 3)
        :return torch.Tensor with shape (T, 3, 224, 224)
        """
        return torch.cat([self.transforms(img).unsqueeze(dim=0) for img in video], dim=0)


if __name__ == '__main__':
    data = TACoS(textual_data_path='data/textual_data/TACoS', visual_data_path='data/processed_visual_data/TACoS',
                 num_time_scales=10, delta=4, threshold=1.)