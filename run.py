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
    --num-time-scales                       Parameter K in the paper
    --time-scale                            Parameter ẟ in the paper
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 20]
"""

import torch
import torch.nn as nn
from models.tgn import TGN
from docopt import docopt
from typing import Dict
from vocab import Vocab
from utils import load_word_vectors, read_corpus
import numpy as np
import sys
from data import NSGVDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_BCE_weights(dataset: NSGVDataset, num_time_scales: int):

    print('Calculating Binary Cross Entropy weights w0, w1...')

    w0 = torch.zeros([num_time_scales, ], dtype=torch.float32)
    w1 = torch.zeros([num_time_scales, ], dtype=torch.float32)

    num_samples = len(dataset)
    T = dataset[0][2].shape[0]

    for i in tqdm(range(num_samples)):
        _, _, y = dataset[i]  # Tensor with shape (T, K)
        tmp = torch.sum(y, dim=0)
        w0 += 1 - tmp
        w1 += tmp

    w0 = w0 / (num_samples * T)
    w1 = w1 / (num_samples * T)
    return w0, w1


def train(vocab: Vocab, args: Dict):
    n_iter = int(args['--n-iter'])
    valid_niter = int(args['--valid-niter'])
    train_sents = read_corpus(args['--src-sents'])
    dev_sents = read_corpus(args['--dev-'])
    batch_size = int(args['--batch_size'])
    delta = int(args['--time-scale'])
    num_time_scales = int(args['--num-time-scales'])

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

    dataset = NSGVDataset(textual_data_path='data/textual_data', visual_data_path='data/visual_data',
                          num_time_scales=10, scale=4)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    w0, w1 = find_BCE_weights(dataset, num_time_scales)  # Tensors with shape (K,)

    for iteration in range(n_iter):
        for i_batch, (visual_input, textual_input, y) in enumerate(dataset):
            optimizer.zero_grad()
            probs = model(textual_input, visual_input)  # shape: (n_batch, T, K)
            loss = -torch.sum(y * w0 * torch.log(probs) + w1 * (1 - y) * torch.log(1 - probs))
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    args = docopt(__doc__)
    words, word_vectors = load_word_vectors('data/glove.840B.300d.txt')
    vocab = Vocab(words)
    train(vocab, args)
