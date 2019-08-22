"""
run.py: Run the Temporally Grounding Network (TGN) model

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]

Options:
    -h --help                               show this screen.
    --train-src-sents=<file>                train sentences
    --dev-src-sents=<file>                  dev source sentences
    --batch-size=<int>                      batch size [default: 64]
    --hidden-size-textual-lstm=<int>        hidden size of textual lstm [default: 512]
    --hidden-size-visual-lstm=<int>         hidden size of visual lstm [default: 512]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.tgn import TGN
from docopt import docopt
from typing import Dict


def train(model: TGN, args: Dict):
    n_epochs = args['--max-epoch']
    valid_niter = args['--valid-niter']



if __name__ == '__main__':
    args = docopt(__doc__)
    print(args[''])

