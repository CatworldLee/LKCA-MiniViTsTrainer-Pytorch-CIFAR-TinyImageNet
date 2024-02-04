from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
# import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from models.create_model import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0


def init_parser():
    parser = argparse.ArgumentParser(description='Small dataset vit quick training script')

    # Data args
    parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
    
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'SVHN', 'FLOWER'], type=str, help='Image Net dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 5e-2)')

    parser.add_argument('--model', type=str, default='vit')

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_false', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    
    parser.add_argument('--resume', default=False, help='Version')
       
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    
    parser.add_argument('--cm',action='store_false' , help='Use Cutmix')
    
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    
    parser.add_argument('--mu',action='store_false' , help='Use Mixup')
    
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    
    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    
    return parser


def main(args):
    global best_acc1    
    
    torch.cuda.set_device(args.gpu)

    data_info = datainfo(logger, args)

    with torch.cuda.device(0):
        model = create_model(data_info['img_size'], data_info['n_classes'], args)
        macs, params = get_model_complexity_info(model, (3, data_info['img_size'], data_info['img_size']), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
logger = log.getLogger(__name__)
parser = init_parser()
args = parser.parse_args()
main(args)
        