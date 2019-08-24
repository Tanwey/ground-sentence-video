import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import csv
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer


class NSGVDataset(torch.utils.data.Dataset):
    def __init__(self, textual_data_path: str, visual_data_path: str, num_time_scales: int, scale: int):
        """
        :param textual_path: directory containing the annotations
        :param visual_path: directory containing the videos
        :param num_time_scales: K in the paper
        :param scale: ẟ in the paper
        """
        super(NSGVDataset, self).__init__()
        self.num_time_scales = num_time_scales
        self.scale = scale
        self.textual_data_path = textual_data_path
        self.visual_data_path = visual_data_path

        files = os.listdir(textual_data_path)

        self.annotations = []
        tokenizer = ToktokTokenizer()

        for file in files:
            with open(os.path.join(textual_data_path, file)) as tsvfile:
                video_id = file.replace('.aligned.tsv', '')
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    start_frame, end_frame = row[0], row[1]
                    sents = set([sent for sent in row[6:] if len(sent) > 0])
                    sents = [tokenizer.tokenize(sent.lower()) for sent in sents]
                    self.annotations += [(video_id, start_frame, end_frame, sent) for sent in sents]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):

        video_id = self.annotations[item][0]
        visual_data = np.load(os.path.join(self.visual_data_path, video_id + '.npy'))  #TODO

        visual_data_tensor = torch.from_numpy(visual_data)

        return (visual_data_tensor, self.annotations[item])



if __name__ == '__main__':
    data = NSGVDataset(textual_data_path='data/textual_data', visual_data_path='data/visual_data',
                       num_time_scales=10, scale=4)

    print(data.annotations[30:35])
