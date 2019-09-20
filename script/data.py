import torch
import os
import csv
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from utils import compute_overlap
from math import ceil
from typing import List
from collections import namedtuple
from utils import pad_labels
import json
import h5py


np.random.seed(42)


TOKENIZER = ToktokTokenizer()
Caption = namedtuple('Caption', ['video_id', 'start_time', 'end_time', 'sent'])


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
        self.name = 'TACoS'
        self.K = K
        self.delta = delta
        self.textual_data_path = textual_data_path
        self.visual_data_path = visual_data_path
        self.threshold = threshold
        self.fps = 30  # number of frames per second for TACoS dataset
        self.sample_rate = self.fps * 5  # note that we sample one frame every five seconds

        files = os.listdir(textual_data_path)

        self.captions = []
        
        # loading the textual data
        for file in files:
            with open(os.path.join(textual_data_path, file)) as tsvfile:
                video_id = file.replace('.aligned.tsv', '')
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    start_frame, end_frame = int(row[0]), int(row[1])
                    sents = set([sent for sent in row[6:] if len(sent) > 0])
                    sents = [TOKENIZER.tokenize(sent.lower()) for sent in sents]
                    self.captions += [Caption(video_id=video_id, start_time=start_frame / self.fps, 
                                              end_time=end_frame / self.fps, sent=sent) 
                                      for sent in sents]

        index_array = list(range(len(self.captions)))
        np.random.shuffle(index_array)

        val_size = int(val_ratio * len(index_array))
        test_size = int(test_ratio * len(index_array))

        self.val_indices = index_array[:val_size]
        self.val_captions = [self.captions[idx] for idx in self.val_indices]
        
        self.test_indices = index_array[val_size:val_size+test_size]
        self.test_captions = [self.captions[idx] for idx in self.test_indices]
        
        self.train_indices = index_array[val_size+test_size:]
        self.test_captions = [self.captions[idx] for idx in self.train_indices]
        
        visual_feature, _, _ = self[0]
        self.visual_feature_size = visual_feature.shape[1]
        print(self.visual_feature_size)

#         print('validation set size is %d' % val_size, file=sys.stderr)
#         print('test set size is %d' % test_size, file=sys.stderr)
#         print('train set size is %d' % len(self.train_indices), file=sys.stderr)

    def __len__(self):
        return len(self.train_captions)

    def _generate_labels(self, visual_data: List[torch.Tensor], captions: List[Caption]):
        """
        :param visual_data: a list of extracted features from videos
        :param textual_data: list of annotations
        :returns labels for the samples
        """

        labels = []
        for v, s in zip(visual_data, captions):
            T = v.shape[0]
            label = torch.zeros([T, self.K], dtype=torch.int32)
            start_time, end_time = s.start_time, s.end_time

            for t in range(T):
                for k in range(self.K):
                    if (compute_overlap((t - (k+1) * self.delta) * self.sample_rate / self.fps,
                                        t * self.sample_rate / self.fps, 
                                        start_time, end_time)) > self.threshold:
                        label[t, k] = 1

            labels.append(label)

        return pad_labels(labels)  # torch.Tensor with shape (num_labels, T, K)

    def __getitem__(self, item):
        """
        :param item: The index of the training instance to be retrieved
        :returns the features of the corresponding video as a torch.Tensor, the 
        caption and the ground-truth label 
        """
        s = self.captions[self.train_indices[item]]
        file = s.video_id + '_features.pt'
        path = os.path.join(self.visual_data_path, file)
        visual_feature = torch.load(path)
        label = self._generate_labels([visual_feature], [s])[0]
        return visual_feature, s, label

    def _load_visual_data(self, captions: List[Caption]):
        """Loads the computed features of TACoS videos correpsonding to a given 
        set of captions
        :param textual_data:
        :return: list of the videos corresponding to the captions
        """
        visual_data = []
        for t in captions:
            video_id = t.video_id
            file = video_id + '_features.pt'
            path = os.path.join(self.visual_data_path, file)
            visual_feature = torch.load(path)
            visual_data.append(visual_feature)

        return visual_data

 
    def data_iter(self, batch_size: int, which_set: str):
        """Iterates over the train, validation, or the test set
        :param batch_size
        :param which_set: specifies the part of the data to iterate over as a 
        str ('train', 'val', or 'test')
        :return: batches of data
        Note that ground truth labels are just used during training procedure.
        """
        index_array = None
        if which_set == 'train':
            index_array = self.train_indices
        elif which_set == 'val':
            index_array = self.val_indices
        elif which_set == 'test':
            index_array = self.test_indices

        batch_num = ceil(len(index_array) / batch_size)
        
        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            batch_captions = [self.captions[idx] for idx in indices]
            batch_captions = sorted(batch_captions, key=lambda s: len(s.sent), reverse=True)

            visual_data = self._load_visual_data(batch_captions)

            if which_set == 'train':
                labels = self._generate_labels(visual_data=visual_data, captions=batch_captions)
                captions_sents = [s.sent for s in batch_captions]
                yield captions_sents, visual_data, labels
            else:
                yield batch_captions, visual_data

    
class ActivityNet(torch.utils.data.Dataset):
    """ActivityNet Captions dataset"""
    def __init__(self, textual_data_path: str, visual_data_path: str, K: int, delta: int, 
                 threshold: float):
        super(ActivityNet, self).__init__()
        
        self.name = 'ActivityNet'
        self.K = K
        self.delta = delta
        self.textual_data_path = textual_data_path
        self.visual_data_path = visual_data_path
        self.threshold = threshold
        self.fps = 30  # number of frames per second in videos
        
        '''
        the provided C3D features have been extracted every 8 frames. In this code 
        I will further sample one frame out of every three frames (Please see the implementation 
        of _load_visual_data) which leads to one frame in every 24 frames of the original video 
        (which gives us roughly one frame per second)
        '''
        self.sample_rate = 24  
        
        self.train_captions = []
        
        with open(os.path.join(textual_data_path, 'train.json')) as json_file:
            data = json.load(json_file)
            
            for key, value in data.items():
                time_stamps = value['timestamps']
                sents = value['sentences']
                sents = [TOKENIZER.tokenize(sent.lower()) for sent in sents]
                self.train_captions += [Caption(video_id=key, start_time=time_stamp[0], 
                                                end_time=time_stamp[1], sent=sent) 
                                        for time_stamp, sent in zip(time_stamps, sents)]

        np.random.shuffle(self.train_captions)
#         print('number of train instances', len(self.train_captions))
        
        self.val_captions = []
        
        with open(os.path.join(textual_data_path, 'val_1.json')) as json_file:
            data = json.load(json_file)
            
            for key, value in data.items():
                time_stamps = value['timestamps']
                sents = value['sentences']
                sents = [TOKENIZER.tokenize(sent.lower()) for sent in sents]
                self.val_captions += [Caption(video_id=key, start_time=time_stamp[0], 
                                              end_time=time_stamp[1], sent=sent) 
                                      for time_stamp, sent in zip(time_stamps, sents)]
        
#         print('number of val instances', len(self.val_captions))
        
        self.test_captions = []
        
        self.visual_features = h5py.File(os.path.join(visual_data_path, 
                                                      'sub_activitynet_v1-3.c3d.hdf5'), 'r')
        
        visual_feature, _, _ = self[0]
        self.visual_feature_size = visual_feature.shape[1]
    
    def __len__(self):
        return len(self.train_captions)
    
    def __getitem__(self, item):
        """
        :param item: The index of the training instance to be retrieved
        :returns the features of the corresponding video as a torch.Tensor, the caption 
        and the ground-truth label 
        """
        s = self.train_captions[item]
        visual_feature = torch.from_numpy(self.visual_features[s.video_id]['c3d_features'][:]).float()
        visual_feature = visual_feature[list(range(0, len(visual_feature), 3))]
        label = self._generate_labels([visual_feature], [s])[0]
        return visual_feature, s, label
    
    def _generate_labels(self, visual_data: List[torch.Tensor], captions: List[Caption]):
        """Generates labels for a given set of captions and videos
        :param visual_data: a list of extracted features as torch.Tensor from videos
        :param textual_data: list of annotations
        :returns labels for the samples as a torch.Tensor with the 
        shape (num_labels, max_time_steps, K)
        """

        labels = []
        for v, s in zip(visual_data, captions):
            T = v.shape[0]
            label = torch.zeros([T, self.K], dtype=torch.int32)
            start_time, end_time = s.start_time, s.end_time

            for t in range(T):
                for k in range(self.K):
                    if (compute_overlap((t - (k+1) * self.delta) * self.sample_rate / self.fps,
                                        t * self.sample_rate / self.fps, 
                                        start_time, end_time)) > self.threshold:
                        label[t, k] = 1

            labels.append(label)
        
        return pad_labels(labels) 
    
    def _load_visual_data(self, captions: List[Caption]):
        """Loads the C3D features of the videos corresponding to the given captions
        :param captions: captions as a list 
        :returns visual_data: list of the videos corresponding to the captions
        """
        visual_data = []
        for t in captions:
            video_id = t.video_id
            visual_feature = torch.from_numpy(self.visual_features[video_id]['c3d_features'][:]).float()
            
            # sampling one third of the frames
            visual_feature = visual_feature[list(range(0, visual_feature.shape[0], 3))]
            visual_data.append(visual_feature)

        return visual_data
    
    def data_iter(self, batch_size: int, which_set: str):
        """Iterates over the train, validation, or the test set
        :param batch_size
        :param which_set: specifies the part of the data to iterate over as a 
        str ('train', 'val', or 'test')
        return: batches of data
        Note that ground truth labels are just used during training procedure.
        """
        if which_set == 'train':
            data = self.train_captions
        elif which_set == 'val':
            data = self.val_captions
        
        indices = list(range(len(data)))
        batch_num = ceil(len(data) / batch_size)
        
        for i in range(batch_num):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            batch_captions = [data[idx] for idx in batch_indices]
            batch_captions = sorted(batch_captions, key=lambda s: len(s.sent), reverse=True)
            
            visual_data = self._load_visual_data(batch_captions)  # a list of torch.Tensors

            if which_set == 'train':
                labels = self._generate_labels(visual_data=visual_data, captions=batch_captions)
                captions_sents = [s.sent for s in batch_captions]   
                yield captions_sents, visual_data, labels
            else:
                yield batch_captions, visual_data
            
