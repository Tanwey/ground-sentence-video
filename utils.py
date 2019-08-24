import torch
from torch.nn import Embedding
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from typing import List
import os
import cv2
import math


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


def load_word_vectors(path):
    print('Loading Glove 300-d word vectors...')
    glove_file = datapath(path)
    word2vec_glove_file = get_tmpfile("glove.word2vec.txt")
    glove2word2vec(glove_file, word2vec_glove_file)
    model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
    words = list(model.vocab.keys())
    word_vectors = np.concatenate([model[word] for word in words])

    return words, word_vectors


def process_visual_data():
    visual_data_path = 'data/visual_data'
    processed_visual_data_path = 'data/processed_visual_data'

    if not os.path.exists(processed_visual_data_path):
        os.mkdir(processed_visual_data_path)

    video_files = os.listdir(visual_data_path)

    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(visual_data_path, video_file))
        success = 1
        frames = []

        current_frame = 0
        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))

        while success:
            if current_frame % 100 == 0:
                print('Processing frame number %d' % current_frame)

            success, frame = cap.read()
            if success:
                if current_frame % fps == 0:
                    frames.append(np.expand_dims(frame, axis=0))

            current_frame += 1

        frames = np.concatenate(frames)
        output_file = os.path.join(processed_visual_data_path, video_file.replace('.avi', '.npy'))
        np.save(output_file, frames)


if __name__ == '__main__':
    process_visual_data()
