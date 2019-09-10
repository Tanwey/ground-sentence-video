"""
run.py: Run the Temporally Grounding Network (TGN) model

Usage:
    run.py train --textual-data-path=<dir> --visual-data-path=<dir> [options]
    run.py test --textual-data-path=<file> --visual-data-path=<file> [options]

Options:
    -h --help                               show this screen
    --textual-data-path=<dir>               directory containing the annotations
    --visual-data-path=<dir>                directory containing the videos
    --batch-size=<int>                      batch size [default: 64]
    --hidden-size-textual-lstm=<int>        hidden size of textual lstm [default: 512]
    --hidden-size-visual-lstm=<int>         hidden size of visual lstm [default: 512]
    --hidden-size-ilstm=<int>               hidden size of ilstm [default: 512]
    --log-every=<int>                       log every [default: 10]
    --n-iter=<int>                          number of iterations of training [default: 200]
    --lr=<float>                            learning rate [default: 0.001]
    --num-time-scales=<int>                 Parameter K in the paper
    --delta=<int>                           Parameter ẟ in the paper
    --threshold=<float>                     Parameter θ in the paper
    --valid-niter=<int>                     perform validation after how many iterations [default: 20]
    --word-embed-size=<int>                 size of the glove word vectors [default: 50]
"""

import torch
import torch.nn as nn
from models.tgn import TGN
from docopt import docopt
from typing import Dict
from vocab import Vocab
from utils import load_word_vectors, pad_visual_data, pad_textual_data
import numpy as np
import sys
from data import TACoS
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from torch.nn.init import xavier_normal_, normal_


def find_bce_weights(dataset: TACoS, num_time_scales: int):
    print('Calculating Binary Cross Entropy weights w0 and w1...', file=sys.stderr)
    w0 = torch.zeros([num_time_scales, ], dtype=torch.float32)

    num_samples = len(dataset)
    time_steps = 0

    for i in tqdm(range(num_samples)):
        _, _, label = dataset[i]
        T = label.shape[0]
        time_steps += T
        tmp = torch.sum(label, dim=0).to(torch.float32)
        w0 += T - tmp

    w0 = w0 / time_steps

    return w0, 1-w0


def eval(model: TGN, valloader: DataLoader, batch_size: int):
    was_training = model.training

    # with torch.no_grad():
    #     for batch_idx, (visual_data, textual_data, y) in enumerate(valloader):
    #         probs = model(visual_data, textual_data)
    #         loss = -torch.sum(y * w0 * torch.log(probs) + w1 * (1 - y) * torch.log(1 - probs))
    #         loss.item()
    #
    # if was_training:
    #     model.train()
    #
    # return loss


def train(vocab: Vocab, word_vectors: np.ndarray, args: Dict):
    n_iter = int(args['--n-iter'])
    valid_niter = int(args['--valid-niter'])
    textual_data_path = args['--textual-data-path']
    visual_data_path = args['--visual-data-path']
    batch_size = int(args['--batch-size'])
    delta = int(args['--delta'])
    num_time_scales = int(args['--num-time-scales'])
    lr = float(args['--lr'])
    log_every = int(args['--log-every'])
    threshold = float(args['--threshold'])

    embedding = nn.Embedding(len(vocab), word_vectors.shape[1], padding_idx=vocab.word2id['<pad>'])
    embedding.weight = nn.Parameter(data=torch.from_numpy(word_vectors).to(torch.float32), requires_grad=False)

    model = TGN(args)
    model.train()

    for p in model.parameters():
        if p.requires_grad:
            if len(p.data.shape) > 1:
                xavier_normal_(p.data)
            else:
                normal_(p.data)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)
    embedding.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = TACoS(textual_data_path=textual_data_path, visual_data_path=visual_data_path,
                    num_time_scales=num_time_scales, delta=delta, threshold=threshold)

    #writer = SummaryWriter()
    #writer.add_graph(model, )

    w0, w1 = find_bce_weights(dataset, num_time_scales)  # Tensors with shape (K,)

    cum_samples = reported_samples = 0
    train_time = begin_time = time()
    print('Begin training...')

    for iteration in range(n_iter):

        # getting visual_data, textual_data, labels each one as a list
        textual_data, visual_data, y = next(dataset.data_iter(batch_size, 'train'))
        lengths_t = [len(t) for t in textual_data]
        textual_data_tensor = vocab.to_input_tensor(textual_data, device=device)  # tensor with shape (n_batch, N)
        textual_data_embed_tensor = embedding(textual_data_tensor)  # tensor with shape (n_batch, N, embed_size)

        optimizer.zero_grad()
        probs, mask = model(textual_input=textual_data_embed_tensor, visual_input=visual_data, lengths_t=lengths_t)  # shape: (n_batch, T, K)
        loss = -torch.sum((y * w0 * torch.log(probs) + w1 * (1 - y) * torch.log(1 - probs))*mask)
        cum_samples += batch_size
        reported_samples += batch_size

        loss.backward()
        optimizer.step()

        if iteration % log_every == 0:
            print('Iteration number %d, loss train: %f, '
                  'speed %.2f samples/sec, time elapsed %.2f sec' % (iteration,
                                                                loss.item(),
                                                                reported_samples / (time() - train_time),
                                                                time() - begin_time))

            reported_samples = 0
            train_time = time()
            #writer.add_scalar('Loss/train', loss_train.item(), iteration)

        # if iteration % valid_niter == 0:
        #     loss_val = eval(dataset, batch_size=batch_size)
            #writer.add_scalar('Loss/val', loss_val)

        #writer.close()


if __name__ == '__main__':
    args = docopt(__doc__)
    word_embed_size = int(args['--word-embed-size'])
    words, word_vectors = load_word_vectors('glove.6B.{}d.txt'.format(word_embed_size))

    # with open('vocab.txt', 'r') as f:
    #    words = f.readlines()
    # print(len(words))
    # word_vectors = np.zeros([len(words)+2, 50])

    vocab = Vocab(words)
    train(vocab, word_vectors, args)
