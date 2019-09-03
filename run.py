"""
run.py: Run the Temporally Grounding Network (TGN) model

Usage:
    run.py train --textual-data-path=<file> --visual-data-path=<file> [options]
    run.py test --textual-data-path=<file> --visual-data-path=<file> [options]
Options:
    -h --help                               show this screen.
    --textual-data-path=<file>              directory containing the annotations
    --visual-data-path=<file>               directory containing the videosakhe gheyre unam mishe
    --batch-size=<int>                      batch size [default: 64]
    --hidden-size-textual-lstm=<int>        hidden size of textual lstm [default: 512]
    --hidden-size-visual-lstm=<int>         hidden size of visual lstm [default: 512]
    --log-every=<int>                       log every [default: 10]
    --n-iter=<int>                          number of iterations of training [default: 200]
    --lr=<float>                            learning rate [default: 0.001]
    --num-time-scales=<int>                 Parameter K in the paper
    --delta=<int>                           Parameter ẟ in the paper
    --threshold=<float>                     Parameter θ in the paper
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 20]
"""

import torch
import torch.nn as nn
from models.tgn import TGN
from docopt import docopt
from typing import Dict
from vocab import Vocab
from utils import load_word_vectors, read_corpus, pad_visual_data, pad_textual_data
import numpy as np
import sys
from data import NSGVDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_bce_weights(dataset: NSGVDataset, num_time_scales: int):
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


def eval(model: TGN, valloader: DataLoader, batch_size: int):
    was_training = model.training

    with torch.no_grad():
        for batch_idx, (visual_data, textual_data, y) in enumerate(valloader):
            probs = model(visual_data, textual_data)
            loss = -torch.sum(y * w0 * torch.log(probs) + w1 * (1 - y) * torch.log(1 - probs))
            loss.item()

    if was_training:
        model.train()


def train(vocab: Vocab, args: Dict):
    n_iter = int(args['--n-iter'])
    valid_niter = int(args['--valid-niter'])
    textual_data_path = args['--textual-data-path']
    visual_data_path = args['--visual-data-path']
    batch_size = int(args['--batch_size'])
    delta = int(args['--delta'])
    num_time_scales = int(args['--num-time-scales'])
    lr = float(args['--lr'])
    log_every = int(args['--log-every'])
    threshold = float(args['--log-every'])

    embedding = nn.Embedding(len(vocab), word_vectors.shape[1], padding_idx=vocab.word2id['<pad>'])
    model = TGN(args)

    model.train()

    for p in model.parameters():
        p.data.xavier_normal_()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = NSGVDataset(textual_data_path=textual_data_path, visual_data_path=visual_data_path,
                          num_time_scales=num_time_scales, delta=delta, threshold=threshold)

    #num_data = len(dataset)
    #num_train, num_val = int(num_data * 0.9), int(num_data * 0.005)
    #train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    w0, w1 = find_bce_weights(dataset, num_time_scales)  # Tensors with shape (K,)

    for iteration in range(n_iter):

        # getting visual_data, textual_data, labels each one as a list
        visual_data, textual_data, y = next(dataset.data_iter(batch_size, 'train'))

        visual_data_tensor = pad_visual_data(visual_data)  # tensor with shape (n_batch, T, 224, 224, 3)
        textual_data_padded = pad_textual_data(textual_data, vocab.word2id['<pad>'])

        print(textual_data_padded.shape)

        optimizer.zero_grad()
        probs = model(textual_data, visual_data)  # shape: (n_batch, T, K)
        loss = -torch.sum(y * w0 * torch.log(probs) + w1 * (1 - y) * torch.log(1 - probs))
        loss.backward()
        optimizer.step()

        if iteration % log_every == 0:
            pass

        if iteration % valid_niter == 0:
            eval(val_loader, batch_size=batch_size)


if __name__ == '__main__':
    args = docopt(__doc__)
    words, word_vectors = load_word_vectors('glove.840B.300d.txt')
    vocab = Vocab(words)
    train(vocab, args)
