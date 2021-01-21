import argparse
import numpy as np
import os
import random


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument("--dataset", type=str, default='syn3', help="Dataset string")# 'cora', 'citeseer', 'pubmed'
    parser.add_argument('--early_stop', type=int, default= 2000, help='early_stop')
    parser.add_argument('--dtype', type=str, default='float32')  #
    parser.add_argument('--seed',type=int, default=12345, help='seed')
    parser.add_argument('--setting', type=int, default=1)  # 1 is used for GNNExplainer paper.

    parser.add_argument('--normadj', type=bool, default=False)  # GNNExplianer doesnot norm adj
    parser.add_argument('--normfea', type=bool, default=False)  # GNNExplianer doesnot norm fea


    parser.add_argument('--motif', type=str, default='house')  #

    parser.add_argument('--order', type=str, default='AW')  #
    parser.add_argument('--embnormlize', type=bool, default=True)  #
    parser.add_argument('--bias', type=bool, default=True)  #


    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--dropout',type=float, default=0.0, help='dropout rate (1 - keep probability).')
    parser.add_argument('--hiddens', type=str, default='20-20-20')
    parser.add_argument("--lr", type=float, default=0.001,help='initial learning rate.')
    parser.add_argument('--bn', type=bool, default=True)
    parser.add_argument('--concat', type=bool, default=True)
    parser.add_argument('--valid', type=bool, default=False)

    parser.add_argument("--elr", type=float, default=0.003,help='initial learning rate.')
    parser.add_argument('--eepochs', type=int, default=20, help='Number of epochs to train explainer.')

    parser.add_argument('--act', type=str, default='relu', help='activation funciton')  #
    parser.add_argument('--initializer', default='glorot')

    parser.add_argument('--save_model',type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='./checkpoints/gcn')

    # ---------------------paramerters for explainers----------------------
    parser.add_argument('--budget', type=float, default=-1.0, help='coefficient for size constriant')

    parser.add_argument('--coff_size', type=float, default=0.0001, help='coefficient for size constriant')
    parser.add_argument('--coff_lap', type=float, default=0., help='coefficient for laplacian')
    parser.add_argument('--coff_ent', type=float, default=0.01, help='coefficient for entropy loss')
    parser.add_argument('--weight_decay',type=float, default=0.0, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--miGroudTruth',type=bool, default=True, help='Mutual Information between hat y and GroundTruth Label')

    parser.add_argument('--coff_t0', type=float, default=5.0, help='initial temperature')
    parser.add_argument('--coff_te', type=float, default=4.99, help='final temperature')
    parser.add_argument('--sample_bias',type=float, default=0.0, help='bias for sampling from 0-1')


    args, _ = parser.parse_known_args()

    return args

args = get_params()
params = vars(args)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf

seed = args.seed
if seed>0:
    random.seed(args.seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

dtype = tf.float32
if args.dtype=='float64':
    dtype = tf.float64

eps = 1e-7
