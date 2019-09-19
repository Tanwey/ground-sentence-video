"""
evaluate.py: Evaluate the TGN model

Usage:
    evaluate.py tacos --model-path=<file> --textual-data-path=<dir> --visual-data-path=<dir> [options]
    evaluate.py acnet --model-path=<file> --textual-data-path=<dir> --visual-data-path=<dir> [options]

Options:
    -h --help                               show this screen
    --textual-data-path=<dir>               directory containing the annotations
    --visual-data-path=<dir>                directory containing the videos
    --K=<int>                               parameter K in the paper
    --delta=<int>                           parameter ẟ in the paper
    --threshold=<float>                     parameter θ in the paper
    --batch-size=<int>                      batch size [default: 32]
    --model-path=<file>                     model load path [default: model.bin]
    --top-n-eval=<int>                      Parameter N in R@N, IOU=θ evaluation metric [default: 1]
"""

import torch
import torch.nn as nn
from models.tgn import TGN
from docopt import docopt
from script.vocab import Vocab
from script.utils import load_word_vectors
from run import  top_n_iou
import sys
from script.data import TACoS, ActivityNet
from tqdm import tqdm
from math import ceil


def evaluate(model: TGN, dataset, embedding: nn.Embedding, K, threshold: float, delta: int, batch_size: int):
    batch_size = int(args['--batch-size'])

    with torch.no_grad():
        cum_score = cum_samples = 0

        pbar = tqdm(total=ceil(len(dataset.val_captions) / batch_size))

        for textual_data, visual_data in iter(dataset.data_iter(batch_size, 'test')):
            cum_samples += len(textual_data)
            lengths_t = [len(t) for t in textual_data]
            sents = [t.sent for t in textual_data]
            textual_data_tensor = vocab.to_input_tensor(sents, device=device)  # tensor with shape (n_batch, N)
            textual_data_embed_tensor = embedding(textual_data_tensor)  # tensor with shape (n_batch, N, embed_size)

            probs, mask = model(visual_data, textual_data_embed_tensor, lengths_t)  # Tensors with shape (n_batch, T, K)

            gold_start_times = [t.start_time for t in textual_data]
            gold_end_times = [t.end_time for t in textual_data]

            score = top_n_iou(probs*mask, gold_start_times, gold_end_times, args, dataset.fps, dataset.sample_rate)
            cum_score += score
            pbar.update()

        pbar.close()
    
    print('test score %.2f' % (cum_score / cum_samples))


if __name__ == '__main__':
    args = docopt(__doc__)
    
    textual_data_path = args['--textual-data-path']
    visual_data_path = args['--visual-data-path']
    batch_size = int(args['--batch-size'])
    delta = int(args['--delta'])
    K = int(args['--K'])
    threshold = float(args['--threshold'])

    word_embed_size=50
    words, word_vectors = load_word_vectors('glove.6B.{}d.txt'.format(word_embed_size))
    vocab = Vocab(words)

    if args['tacos']:
        dataset = TACoS(textual_data_path=textual_data_path, visual_data_path=visual_data_path, 
                        K=K, delta=delta, threshold=threshold)
    elif args['acnet']:
        dataset = ActivityNet(textual_data_path=textual_data_path, visual_data_path=visual_data_path, 
                              K=K, delta=delta, threshold=threshold)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('use device: %s' % device, file=sys.stderr)
    
    print('loading the model from %s ...' % args['--model-path'])
    model = TGN.load(args['--model-path'])
    model.to(device)
    
    embedding = nn.Embedding(len(vocab), word_vectors.shape[1], padding_idx=vocab.word2id['<pad>'])
    embedding.weight = nn.Parameter(data=torch.from_numpy(word_vectors).to(torch.float32), requires_grad=False)
    embedding.to(device)
    
    evaluate(model, dataset, embedding, K, threshold, delta, batch_size)