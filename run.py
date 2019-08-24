"""
run.py: Run the Temporally Grounding Network (TGN) model

Usage:
    run.py train --train-sents=<file> --train-imgs=<file> --dev-sents=<file> [options]

Options:
    -h --help                               show this screen.
    --train-sents=<file>                    train sentences
    --dev-sents=<file>                      dev source sentences
    --batch-size=<int>                      batch size [default: 64]
    --hidden-size-textual-lstm=<int>        hidden size of textual lstm [default: 512]
    --hidden-size-visual-lstm=<int>         hidden size of visual lstm [default: 512]
    --log-every=<int>                       log every [default: 10]
    --n-iter=<int>                          number of iterations of training [default: 200]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 20]
"""


import torch
import torch.nn as nn
from tqdm import tqdm
from models.tgn import TGN
from docopt import docopt
from typing import Dict
from vocab import Vocab
from utils import load_word_vectors, read_corpus
import numpy as np
import sys


def train(vocab: Vocab, args: Dict):
    n_iter = int(args['--n-iter'])
    valid_niter = int(args['--valid-niter'])
    train_sents = read_corpus(args['--src-sents'])
    dev_sents = read_corpus(args['--dev-'])
    batch_size = int(args['--batch_size'])

    lr = float(args['--lr'])
    log_every = args['--log-every']
    uniform_init = float(args['--uniform-init'])

    embedding = nn.Embedding(len(vocab), word_vectors.shape[1], padding_idx=vocab.word2id['<pad>'])
    model = TGN(args)

    model.train()

    if np.abs(uniform_init) > 0.:
        print('Uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = []  # TODO

    for i in range(n_iter):
        for sents, imgs in dataset:

            optimizer.zero_grad()

            scores = model(sents, imgs)





if __name__ == '__main__':
    args = docopt(__doc__)
    words, word_vectors = load_word_vectors('glove.840B.300d.txt')
    vocab = Vocab(words)
    train(vocab, args)
