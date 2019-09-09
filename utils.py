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
from matplotlib import pyplot as plt
from skimage import transform


class ModelEmbeddings:
    def __init__(self, word_vectors_np, padding_idx=0):
        self.embedding = Embedding(len(word_vectors_np), embedding_dim=300, padding_idx=padding_idx)
        Embedding.weight = torch.from_numpy(word_vectors_np)


def pad_textual_data(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    :param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    :param pad_token (str): padding token

    :returns sents_padded (list[list[str]]): list of sentences where sentences shorter
    than the max length sentence are padded out with the pad_token, such that
    each sentences in the batch now has equal length.
    """
    longest = np.max([len(sent) for sent in sents])
    sents_padded = list(map(lambda sent: sent + [pad_token] * (longest - len(sent)), sents))

    return sents_padded


def pad_visual_data(visual_data: List[torch.Tensor]):
    """
    :param visual_data:
    :return:
    """
    feature_dim = visual_data[0].shape[1]
    max_len = np.max([v.shape[0] for v in visual_data])

    visual_data_padded = list(map(lambda v: torch.cat([v, torch.zeros(max_len - v.shape[0], feature_dim)]).
                                  unsqueeze(dim=0),
                                  visual_data))

    return torch.cat(visual_data_padded, dim=0)  # tensor with shape (n_batch, max_len, feature_dim)


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
    print('Loading Glove word vectors...')

    if not os.path.exists('glove.word2vec.txt'):
        glove2word2vec(glove_file_path, 'glove.word2vec.txt')

    model = KeyedVectors.load_word2vec_format('glove.word2vec.txt')
    words = list(model.vocab.keys())
    dim = len(model[words[0]])
    print('dimension of word vectors', dim)
    word_vectors = [np.zeros([2, dim])] + [model[word].reshape(1, -1) for word in words]
    word_vectors = np.concatenate(word_vectors, axis=0)
    print('shape of word vectors', word_vectors.shape)
    print('Word vectors were loaded successfully...')

    return words, word_vectors


def process_visual_data_tacos(output_frame_size: Tuple):
    visual_data_path = 'data/visual_data/TACoS'
    processed_visual_data_path = 'data/processed_visual_data/TACoS'

    if not os.path.exists(processed_visual_data_path):
        os.mkdir(processed_visual_data_path)

    video_files = os.listdir(visual_data_path)
    if '.DS_Store' in video_files: video_files.remove('.DS_Store')

    f = open('fps.txt', 'w')

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
            else:
                break
            current_frame += 1

        frames = np.concatenate(frames)
        output_file = os.path.join(processed_visual_data_path, video_file.replace('.avi', '.npy'))
        f.write(video_file.replace('.avi', '') + '\t' + str(current_frame))
        np.save(output_file, frames)

    f.close()


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
    pass
