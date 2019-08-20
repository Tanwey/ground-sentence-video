import torch
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec



class ModelEmbeddings:
    def __init__(self, path):
        pass


    def _load_word_vectors(self, path):
        glove_file = datapath(path)
        word2vec_glove_file = get_tmpfile("glove.word2vec.txt")
        glove2word2vec(glove_file, word2vec_glove_file)
        model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
        words = list(model.vocab.keys())



