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
import json
import h5py


Caption = namedtuple('Caption', ['video_id', 'start', 'end', 'sent'])


class TACoS(torch.utils.data.Dataset):
    def __init__(self, textual_data_path: str, visual_data_path: str, delta: int, K: int, 
                 threshold: float, val_ratio=0.25, test_ratio=0.25):
        """
        :param textual_data_path: directory containing the annotations
        :param visual_data_path: directory containing the features extracted from videos
        :param num_time_scales: K in the paper
        :param delta: ẟ in the paper
        :param threshold: θ in the paper
        """
        super(TACoS, self).__init__()
        self.K = K
        self.delta = delta
        self.textual_data_path = textual_data_path
        self.visual_data_path = visual_data_path
        self.threshold = threshold

        files = os.listdir(textual_data_path)

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
                    self.textual_data += [Caption(video_id=video_id, start=start_frame, end=end_frame, 
                                                  sent=sent) for sent in sents]

        np.random.seed(42)
        index_array = list(range(len(self.textual_data)))
        np.random.shuffle(index_array)

        val_size = int(val_ratio * len(index_array))
        test_size = int(test_ratio * len(index_array))

        self.val_indices = index_array[:val_size]
        self.test_indices = index_array[val_size:val_size+test_size]
        self.train_indices = index_array[val_size+test_size:]

        print('Validation set size is %d' % val_size, file=sys.stderr)
        print('Test set size is %d' % test_size, file=sys.stderr)
        print('Train set size is %d' % len(self.train_indices), file=sys.stderr)

    def __len__(self):
        return len(self.textual_data)

    def _generate_labels(self, visual_data: List[torch.Tensor], textual_data: List[Caption]):
        """
        :param visual_data: a list of extracted features from videos
        :param textual_data: list of annotations
        :returns labels for the samples
        """
        # number of frames per second for TACoS dataset
        fps = 30
        
        # note that we sampled one frame every five seconds while we pre-processed the data
        sample_rate = fps * 5

        labels = []
        for v, s in zip(visual_data, textual_data):
            T = v.shape[0]
            label = torch.zeros([T, self.K], dtype=torch.int32)
            start_frame, end_frame = s.start, s.end

            for t in range(T):
                for k in range(self.K):
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
        file = s.video_id + '_features.pt'
        path = os.path.join(self.visual_data_path, file)
        frames = torch.load(path)
        label = self._generate_labels([frames], [s])[0]
        return frames, s, label

    def _load_visual_data(self, textual_data: List[Caption]):
        """
        :param textual_data:
        :return: list of the videos corresponding to the annotations
        """
        visual_data = []
        for t in textual_data:
            video_id = t.video_id
            file = video_id + '_features.pt'
            path = os.path.join(self.visual_data_path, file)
            features = torch.load(path)
            visual_data.append(features)

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

    
class ActivityNet(torch.utils.data.Dataset):
    def __init__(self, textual_data_path: str, visual_data_path: str, K: int, delta: int, threshold: float):
        super(ActivityNet, self).__init__()
        
        self.K = K
        self.delta = delta
        self.textual_data_path = textual_data_path
        self.visual_data_path = visual_data_path
        self.threshold = threshold
            
        tokenizer = ToktokTokenizer()
        
        self.train_captions = []
        
        with open(os.path.join(textual_data_path, 'train.json')) as json_file:
            data = json.load(json_file)
            
            for key, value in data.items():
                time_stamps = value['timestamps']
                sents = value['sentences']
                sents = [tokenizer.tokenize(sent.lower()) for sent in sents]

                self.train_captions += [Caption(video_id=key, start=time_stamp[0], end=time_stamp[1], 
                                                sent=sent) for time_stamp, sent in zip(time_stamps, sents)]
        
        self.val_captions = []
        
        with open(os.path.join(textual_data_path, 'val_1.json')) as json_file:
            data = json.load(json_file)
            for key, value in data.items():
                time_stamps = value['timestamps']
                sents = value['sentences']
                sents = [tokenizer.tokenize(sent.lower()) for sent in sents]
                self.val_captions += [Caption(video_id=key, start=time_stamp[0], end=time_stamp[1], 
                                              sent=sent) for time_stamp, sent in zip(time_stamps, sents)]
        
        self.visual_features = h5py.File(os.path.join(visual_data_path, 'sub_activitynet_v1-3.c3d.hdf5'), 'r')
        
#        feat_video_ith = fid[video_lst[ith]]['c3d_features'][:]
    
    def __len__(self):
        return len(self.train_captions)
    
    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        s = self.train_captions[item]
        visual_feature = torch.from_numpy(self.visual_features[s.video_id]['c3d_features'][:]).float()
        visual_feature = visual_features[list(range(0, len(features), 3))]
        label = self._generate_labels([visual_feature], [s])[0]
        return visual_feature, s, label
    
    def _generate_labels(self, visual_data: List[torch.Tensor], textual_data: List[Caption]):
        """
        :param visual_data: a list of extracted features from videos
        :param textual_data: list of annotations
        :returns labels for the samples
        """
        # number of frames per second for TACoS dataset
        fps = 30
        
        # note that we sampled one frame every five seconds while we pre-processed the data
        sample_rate = fps * 5

        labels = []
        for v, s in zip(visual_data, textual_data):
            T = v.shape[0]
            label = torch.zeros([T, self.K], dtype=torch.int32)
            start_time, end_time = s.start, s.end

            for t in range(T):
                for k in range(self.K):
                    if (compute_overlap((t - (k+1) * self.delta) * sample_rate,
                                        t * sample_rate, start_time, end_time) / fps) > self.threshold:
                        label[t, k] = 1

            labels.append(label)
        
        return pad_labels(labels)
    
    def _load_visual_data(self, textual_data: List[Caption]):
        """
        :param textual_data:
        :return: list of the videos corresponding to the annotations
        """
        visual_data = []
        for t in textual_data:
            video_id = t.video_id
            features = torch.from_numpy(self.visual_features[video_id]['c3d_features'][:]).float()
            features = features[list(range(0, len(features), 3))]  # sample one third of the frames
            visual_data.append(features)

        return visual_data
    
    def data_iter(self, batch_size: int, set: str):
        
        if set == 'train':
            data = self.train_captions
        elif set == 'val':
            data = self.val_captions
        
        indices = list(range(len(data)))
        np.random.shuffle(indices)

        batch_num = ceil(len(data) / batch_size)
        
        for i in range(batch_num):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            textual_data = [data[idx] for idx in batch_indices]
            textual_data = sorted(textual_data, key=lambda s: len(s.sent), reverse=True)
            
            visual_data = self._load_visual_data(textual_data)  # a list of torch.Tensors

            if set == 'train':
                labels = self._generate_labels(visual_data=visual_data, textual_data=textual_data)
                textual_data = [s.sent for s in textual_data]
                yield textual_data, visual_data, labels
            else:
                yield textual_data, visual_data
            
