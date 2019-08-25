import torch
from torch.utils.data.sampler import SubsetRandomSampler
import os
import csv
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from utils import compute_overlap


class NSGVDataset(torch.utils.data.Dataset):
    def __init__(self, textual_data_path: str, visual_data_path: str, num_time_scales: int, delta: int,
                 threshold: float):
        """
        :param textual_path: directory containing the annotations
        :param visual_path: directory containing the videos
        :param num_time_scales: K in the paper
        :param scale: ẟ in the paper
        :param threshold: θ in the paper
        """
        super(NSGVDataset, self).__init__()
        self.num_time_scales = num_time_scales
        self.delta = delta
        self.textual_data_path = textual_data_path
        self.visual_data_path = visual_data_path
        self.threshold = threshold

        files = os.listdir(textual_data_path)

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
        :param item: The index of the sample for which the label is going to be computed
        :return: label with shape (T, K) where T is the length of the visual_input
        """
        T = visual_data.shape[0]
        label = torch.zeros([T, ], dtype=torch.int32)
        for t in range(T):
            for k in range(self.num_time_scales):
                if compute_overlap(t - k * self.delta, t, start_frame, end_frame) > self.threshold:
                    label[t, k] = 1

        return label

    def __getitem__(self, item):

        video_id, start_frame, end_frame = self.textual_data[item][0], self.textual_data[item][1], \
                                           self.textual_data[item][2]
        visual_data = np.load(os.path.join(self.visual_data_path, video_id + '.npy'))  #TODO

        visual_data_tensor = torch.from_numpy(visual_data)

        labels = self._generate_label(visual_data_tensor.shape[0], start_frame, end_frame)

        return visual_data_tensor, self.textual_data[item], labels


if __name__ == '__main__':
    data = NSGVDataset(textual_data_path='data/textual_data', visual_data_path='data/processed_visual_data',
                       num_time_scales=10, delta=4, threshold=1.)  # TODO

    a = data[14329]
    print(a[1])
