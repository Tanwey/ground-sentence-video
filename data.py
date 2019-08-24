import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import csv
import numpy as np


class NSGVDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_path: str, videos_path: str, num_time_scales: int, scale: int):
        """
        :param textual_path: directory containing the annotations
        :param visual_path: directory containing the videos
        :param num_time_scales: K in the paper
        :param scale: ẟ in the paper
        """
        super(NSGVDataset, self).__init__()
        self.num_time_scales = num_time_scales
        self.scale = scale
        self.textual_path = annotations_path
        self.videos_path = videos_path

        files = os.listdir(annotations_path)

        self.annotations = []

        for file in files:
            with open(os.path.join(annotations_path, file)) as tsvfile:
                video_id = file.replace('.aligned.tsv', '')
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    start_frame, end_frame = row[0], row[1]
                    sents = [sent for sent in row[6:] if len(sent)>0]
                    self.annotations += [(video_id, start_frame, end_frame, sent) for sent in sents]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):

        video_id = self.annotations[item]
        visual_data = np.load(os.path.join(self.videos_path, video_id + '.npy'))  #TODO

        visual_data_tensor = torch.from_numpy(visual_data)
        return (visual_data_tensor, self.annotations[item])




