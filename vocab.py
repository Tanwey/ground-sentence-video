from typing import List
import torch
from utils import pad_textual_data
import sys


class Vocab(object):
    """ Vocabulary
    Constructing the vocabulary with the words of Glove
    """
    def __init__(self, words):
        """
        Init VocabEntry Instance.
        :param words: list of words
        """
        print('Creating vocabulary...', file=sys.stderr)
        self.word2id = dict()
        self.word2id['<pad>'] = 0   # Pad Token
        self.word2id['<unk>'] = 1  # Unk token

        for i, word in enumerate(words):
            self.word2id[word] = i+2

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        :param word: word to look up.
        :returns index: index of word
        """
        return self.word2id.get(word, self.word2id['<unk>'])

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        :param word: word to look up
        :returns contains: whether word is contained
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry."""
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        :returns len: number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        :param wid: word index
        :returns word: word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """
        Add word to VocabEntry, if it is previously unseen.
        :param word: word to add to VocabEntry
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words into list or list of list of indices.
        :param sents: (list[str] or list[list[str]] sentence(s) in words
        :return word_ids: (list[int] or list[list[int]]) sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids: List[int]):
        """ Convert list of indices into words.
        :param word_ids: list of word ids
        :returns sents: list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.
        :param sents: list of sentences (words)
        :param device: device on which to load the tesnor, i.e. CPU or GPU
        :returns sents_var: tensor of (batch_size, max_sentence_length)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_textual_data(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return sents_var
