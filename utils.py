"""
Some utility functions

Usage:
    utils.py extract-frames-tacos --visual-data-path=<dir> --processed-visual-data-path=<dir> --output-frame-size=<int>
    utils.py find-K --textual-data-path=<dir>
    utils.py extract-features --frames-path=<dir> --features-path=<dir>

"""


import torch
from torch.nn import Embedding
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from typing import Tuple, List
import os
import cv2
import math
import csv
from torchvision import transforms
from matplotlib import pyplot as plt
from skimage import transform
from docopt import docopt
import sys
import torch.nn as nn
from models.cnn_encoder import VGG16


def pad_textual_data(sents: List[List[str]], pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    :param sents: list of sentences, where each sentence
                                    is represented as a list of words
    :param pad_token: padding token

    :returns sents_padded: list of sentences where sentences shorter
    than the max length sentence are padded out with the pad_token, such that
    each sentences in the batch now has equal length.
    """
    longest = np.max([len(sent) for sent in sents])
    sents_padded = list(map(lambda sent: sent + [pad_token] * (longest - len(sent)), sents))

    return sents_padded


def pad_labels(labels: List[torch.Tensor]):
    """
    :param labels: a list with length num_labels of torch.Tensor
    :returns labels_padded: returns a torch.Tensor with shape (num_labels, T, K)
    """
    num_labels = len(labels)
    max_len = np.max([label.shape[0] for label in labels])
    K = labels[0].shape[1]
    labels_padded = torch.zeros([num_labels, max_len, K])

    for i in range(num_labels):
        labels_padded[i, :labels[i].shape[0], :] = labels[i]

    return labels_padded


def load_word_vectors(glove_file_path):
    print('Loading GloVE word vectors from {}...'.format(glove_file_path), file=sys.stderr)

    if not os.path.exists('glove.word2vec.txt'):
        glove2word2vec(glove_file_path, 'glove.word2vec.txt')

    model = KeyedVectors.load_word2vec_format('glove.word2vec.txt')
    words = list(model.vocab.keys())
    dim = len(model[words[0]])
    word_vectors = [np.zeros([2, dim])] + [model[word].reshape(1, -1) for word in words]
    word_vectors = np.concatenate(word_vectors, axis=0)

    return words, word_vectors


def extract_frames_tacos(visual_data_path: str, processed_visual_data_path: str, output_frame_size: Tuple):

    if not os.path.exists(processed_visual_data_path):
        os.mkdir(processed_visual_data_path)

    video_files = os.listdir(visual_data_path)

    for video_file in video_files:
        print('processing %s...' % video_file)
        cap = cv2.VideoCapture(os.path.join(visual_data_path, video_file))
        success = 1
        frames = []

        current_frame = 0
        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))

        while success:

            success, frame = cap.read()
            if success:
                if current_frame % (fps * 5) == 0:  # sampling one frame every five seconds
                    frame = transform.resize(frame, output_frame_size)  # resize the image
                    frames.append(np.expand_dims(frame, axis=0))
            else:
                break
            current_frame += 1

        frames = np.concatenate(frames).astype(np.float32)
        output_file = os.path.join(processed_visual_data_path, video_file.replace('.avi', '.npy'))
        np.save(output_file, frames)

        
def extract_features(frames_path: str, features_path: str):
    """
    extract the features using the cnn encoder.
    """
    files = os.listdir(frames_path)
    
    # A standard transform needed to be applied to inputs of the models pre-trained on ImageNet
    transform_ = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cnn_encoder = VGG16()
    device = 'cuda:0'
    cnn_encoder.to(device)
    
    for file in files:
        print('Extracting features of %s' % file)
        frames = np.load(os.path.join(preprocessed_visual_data_path, file))
        frames_tensor = torch.cat([transform_(frame).unsqueeze(dim=0) for frame in frames], dim=0)
        features = cnn_encoder(frames_tensor.to(device))
        out_file = os.path.join(features_path, file.replace('.npy', '_features.pt'))
        torch.save(features, out_file)
        

def load_features_activitynet(features_path: str):
    video_ids = list(fid.keys())

    # This line clearly shows that the features are stored as Group/Dataset
    feat_video_ith = fid[video_lst[ith]]['c3d_features'][:]
    
    return 

def extract_frames_didemo():
    pass


def find_K(textual_data_path: str):
    lengths = []
    for file in os.listdir(textual_data_path):
        with open(os.path.join(textual_data_path, file)) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                start_frame, end_frame = int(row[0]), int(row[1])
                lengths.append(end_frame - start_frame)

    print(np.mean(lengths))
    print(np.sort(lengths))
    plt.hist(lengths)
    plt.show()


def compute_overlap(start_a: float, end_a: float, start_b: float, end_b: float):
    """
    :param start_a: start time of first segment
    :param end_a: end frame of first segment
    :param start_b: start frame of second segment
    :param end_b: end frame of second segment
    :return: number of overlapping frames between two segments
    """
    if end_a < start_b or end_b < start_a:
        return 0

    if start_a <= start_b:
        if start_b <= end_a <= end_b:
            return end_a - start_b
        elif end_a > end_b:
            return end_b - start_b
    else:
        if start_a <= end_b <= end_a:
            return end_b - start_a
        elif end_b > end_a:
            return end_a - start_a


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['process-visual-data-tacos']:
        visual_data_path = args['--visual-data-path']
        processed_visual_data_path = args['--processed-visual-data-path']
        output_frame_size = int(args['--output-frame-size'])
        process_visual_data_tacos(visual_data_path, processed_visual_data_path, (output_frame_size, output_frame_size))
    elif args['extract-features']:
        extract_features(args['--preprocessed-visual-data-path'], args['--features-path'])
    elif args['find-K']:
        find_K(args['--textual-data-path'])
