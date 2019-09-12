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
    --max-iter=<int>                        maximum number of iterations of training [default: 200]
    --lr=<float>                            learning rate [default: 0.001]
    --patience=<int>                        waiting for how many iterations to decay learning rate [default: 5]
    --num-time-scales=<int>                 parameter K in the paper
    --delta=<int>                           parameter ẟ in the paper
    --threshold=<float>                     parameter θ in the paper
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --model-save-path=<file>                model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 50]
    --word-embed-size=<int>                 size of the glove word vectors [default: 50]
    --top-n-eval=<int>                      Parameter N in R@N, IOU=θ evaluation metric
"""

import torch
import torch.nn as nn
from models.tgn import TGN
from docopt import docopt
from typing import Dict, List
from vocab import Vocab
from utils import load_word_vectors, pad_visual_data, pad_textual_data
import numpy as np
import sys
from data import TACoS
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from torch.nn.init import xavier_normal_, normal_
from torch.utils.tensorboard import SummaryWriter
from utils import compute_overlap


def top_n_iou(y_pred: torch.Tensor, start_frames: List[int], end_frames: List[int], args: Dict):
    """
    :param y_pred: torch.Tensor with shape (n_batch, T, K)
    :param start_frames: ground truth start frames with len (n_batch,)
    :param end_frames: ground truth end frames with len (n_batch,)
    :returns score: validation score
    """
    n_batch, T, K = y_pred.shape

    delta = int(args['--delta'])
    threshold = float(args['--threshold'])

    # computing indices which is a Tensor with shape (n_batch, top_n_eval)
    _, indices = torch.topk(y_pred.view(n_batch, -1), k=int(args['--top-n-eval']), dim=-1)

    end_time_steps = indices // K  # tensor with shape (n_batch, top_n_eval)
    scale_nums = indices % K
    start_time_steps = end_time_steps - (scale_nums * delta)

    score = 0
    for i in range(n_batch):
        val = np.max([compute_overlap(start_time_step.item(), end_time_step.item(), start_frames[i], end_frames[i])
                        for start_time_step, end_time_step in zip(start_time_steps, end_time_steps)])
        score += int(val > threshold)

    return score


def find_bce_weights(dataset: TACoS, num_time_scales: int, device):
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

    w0 = (w0 / time_steps).to(device)

    return w0, 1-w0


def eval(model: TGN, dataset: TACoS, device, embedding: nn.Embedding, args: Dict):
    was_training = model.training

    batch_size = args['--batch-size']

    with torch.no_grad():
        cum_score = cum_samples = 0

        for textual_data, visual_data, y in iter(dataset.data_iter(batch_size, 'val')):
            cum_samples += len(textual_data)
            lengths_t = [len(t) for t in textual_data]
            textual_data_tensor = vocab.to_input_tensor(textual_data, device=device)  # tensor with shape (n_batch, N)
            textual_data_embed_tensor = embedding(textual_data_tensor)  # tensor with shape (n_batch, N, embed_size)

            probs, mask = model(visual_data, textual_data_embed_tensor, lengths_t)  # Tensors with shape (n_batch, T, K)

            start_frames = [t.start_fram for t in textual_data]
            end_frames = [t.end_frame for t in textual_data]

            score = top_n_iou(probs*mask, start_frames, end_frames, args)
            cum_score += score

    if was_training:
        model.train()

    return cum_score / cum_samples


def train(vocab: Vocab, word_vectors: np.ndarray, args: Dict):
    max_iter = int(args['--max-iter'])
    valid_niter = int(args['--valid-niter'])
    textual_data_path = args['--textual-data-path']
    visual_data_path = args['--visual-data-path']
    batch_size = int(args['--batch-size'])
    delta = int(args['--delta'])
    num_time_scales = int(args['--num-time-scales'])
    lr = float(args['--lr'])
    log_every = int(args['--log-every'])
    threshold = float(args['--threshold'])
    model_save_path = args['--model-save-path']

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

    writer = SummaryWriter()

    w0, w1 = find_bce_weights(dataset, num_time_scales, device)  # Tensors with shape (K,)

    cum_samples = report_samples = 0.
    report_loss = cum_loss = 0.

    val_scores = []
    patience = num_trial = 0

    train_time = begin_time = time()
    print('Begin training...')

    for iteration in range(max_iter):

        # getting visual_data, textual_data, labels each one as a list
        textual_data, visual_data, y = next(dataset.data_iter(batch_size, 'train'))
        lengths_t = [len(t) for t in textual_data]
        textual_data_tensor = vocab.to_input_tensor(textual_data, device=device)  # tensor with shape (n_batch, N)
        textual_data_embed_tensor = embedding(textual_data_tensor)  # tensor with shape (n_batch, N, embed_size)

        optimizer.zero_grad()

        # Computing probs and mask with shape (n_batch, T, K)
        probs, mask = model(textual_input=textual_data_embed_tensor, visual_input=visual_data, lengths_t=lengths_t)

        y = y.to(device)
        batch_loss = -torch.sum((w0 * y * torch.log(probs) + w1 * (1 - y) * torch.log(1 - probs)) * mask)
        batch_loss_val = batch_loss.item()

        cum_samples += batch_size
        report_samples += batch_size
        report_loss += batch_loss_val
        cum_loss += batch_loss_val

        batch_loss.backward()
        optimizer.step()

        if iteration % log_every == 0:
            print('Iteration number %d, loss: %f, '
                  'speed %.2f samples/sec, time elapsed %.2f sec' % (iteration,
                                                                     report_loss / report_samples,
                                                                     report_samples / (time() - train_time),
                                                                     time() - begin_time))

            writer.add_scalar('Loss/train', report_loss/report_samples, iteration)
            report_samples = 0
            report_loss = 0.
            train_time = time()

        if iteration % valid_niter == 0:
            print('Begin Validation...')
            val_score = eval(model=model, dataset=dataset, batch_size=batch_size,
                             device=device, embedding=embedding, top_n_eval=top_n_eval)

            print('Validation score %f' % val_score.item())
            writer.add_scalar('Score/val', val_score.item(), iteration)

            is_better = len(val_scores) == 0 or val_score > np.max(val_scores)
            if is_better:
                patience = 0
                print('Save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                model.save(model_save_path)
                torch.save(optimizer.state_dict(), model_save_path + '.optim')
            elif patience < int(args['--patience']):
                patience += 1
                print('hit patience %d' % patience, file=sys.stderr)

                if patience == int(args['--patience']):
                    num_trial += 1
                    print('hit trial %d' % num_trial, file=sys.stderr)
                    if num_trial == int(args['--max-num-trial']):
                        print('early stop!', file=sys.stderr)
                        exit(0)

                    lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])

                    print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)
                    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                    model.load_state_dict(params['state_dict'])
                    model = model.to(device)

                    print('restore parameters of the optimizers', file=sys.stderr)
                    optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                    # set new lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    patience = 0

    writer.close()


if __name__ == '__main__':
    args = docopt(__doc__)
    word_embed_size = int(args['--word-embed-size'])
    words, word_vectors = load_word_vectors('glove.6B.{}d.txt'.format(word_embed_size))

    # with open('vocab.txt', 'r') as f:
    #   words = f.readlines()
    # print(len(words))
    # word_vectors = np.zeros([len(words)+2, 50])

    vocab = Vocab(words)
    train(vocab, word_vectors, args)
