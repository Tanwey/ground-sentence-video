import torch
from torch.nn import Embedding
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from typing import Tuple, List
import os
import cv2
import math
import csv
from matplotlib import pyplot as plt
from skimage import transform


class ModelEmbeddings:
    def __init__(self, word_vectors_np, padding_idx=0):
        self.embedding = Embedding(len(word_vectors_np), embedding_dim=300, padding_idx=padding_idx)
        Embedding.weight = torch.from_numpy(word_vectors_np)


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    :param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    :param pad_token (str): padding token

    :returns sents_padded (list[list[str]]): list of sentences where sentences shorter
    than the max length sentence are padded out with the pad_token, such that
    each sentences in the batch now has equal length.
    """
    longest = max([len(sent) for sent in sents])
    sents_padded = list(map(lambda sent: sent + [pad_token] * (longest - len(sent)), sents))

    return sents_padded


def read_corpus(file_path):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        data.append(sent)

    return data


def pad_videos():
    pass


def load_word_vectors(path):
    print('Loading Glove 300-d word vectors...')
    glove_file = datapath(path)
    word2vec_glove_file = get_tmpfile("glove.word2vec.txt")
    glove2word2vec(glove_file, word2vec_glove_file)
    model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
    words = list(model.vocab.keys())
    word_vectors = np.concatenate([model[word] for word in words])

    return words, word_vectors


def process_visual_data_tacos(output_frame_size: Tuple):
    visual_data_path = 'data/visual_data/TACoS'
    processed_visual_data_path = 'data/processed_visual_data/TACoS'

    if not os.path.exists(processed_visual_data_path):
        os.mkdir(processed_visual_data_path)

    video_files = os.listdir(visual_data_path)
    if '.DS_Store' in video_files: video_files.remove('.DS_Store')

    for video_file in video_files:
        print('processing %s...' % video_file)
        cap = cv2.VideoCapture(os.path.join(visual_data_path, video_file))
        success = 1
        frames = []

        current_frame = 0
        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
        print('Frame per second is %d' % fps)

        while success:
            if current_frame % 100 == 0:
                print('\t***frame number %d' % current_frame)

            success, frame = cap.read()
            if success:
                if current_frame % (fps * 5) == 0:  # capturing one frame every five seconds
                    frame = transform.resize(frame, output_frame_size)  # resize the image
                    frames.append(np.expand_dims(frame, axis=0))

            current_frame += 1

        frames = np.concatenate(frames)
        output_file = os.path.join(processed_visual_data_path, video_file.replace('.avi', '.npy'))
        np.save(output_file, frames)


def process_visual_data_activitynet():
    pass


def process_visual_data_didemo():
    pass


def find_K():
    textual_data_path = 'data/textual_data'

    lengths = []
    for file in os.listdir(textual_data_path):
        with open(os.path.join(textual_data_path, file)) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                start_frame, end_frame = int(row[0]), int(row[1])
                lengths.append(end_frame - start_frame)

    print(np.mean(lengths))
    plt.hist(lengths)
    plt.show()


def compute_overlap(start_a, end_a, start_b, end_b):
    """
    :param start_a: start frame of first segment
    :param end_a: end frame of first segment
    :param start_b: start frame of second segment
    :param end_b: end frame of second segment
    :return: number of overlapping frames between two segments
    """
    if end_a < start_b or end_b < start_a:
        return 0

    if start_a <= start_b:
        if start_b <= end_a <= end_b:
            return end_a - start_b + 1
        elif end_a > end_b:
            return end_b - start_b + 1
    else:
        if start_a <= end_b <= end_a:
            return end_b - start_a + 1
        elif end_b > end_a:
            return end_a - start_a + 1


if __name__ == '__main__':
    process_visual_data_tacos(output_frame_size=(224, 224))
