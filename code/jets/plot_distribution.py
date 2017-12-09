import numpy as np

import os
import pickle
import logging
import argparse
import datetime
import sys

import smtplib
from email.mime.text import MIMEText

from utils import ExperimentHandler
from loading import load_tf
from loading import load_test

from analysis.distribution import get_prediction

import matplotlib.pyplot as plt

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

parser.add_argument("-d", "--data_list_filename", type=str, default='evaldatasets.txt')
parser.add_argument("-n", "--n_test", type=int, default=-1)
parser.add_argument("-s", "--set", type=str, default='valid')
parser.add_argument("-m", "--model_list_filename", type=str, default='evalmodels.txt')
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-o", "--remove_outliers", action="store_true")
parser.add_argument("-l", "--load_rocs", type=str, default=None)

# logging args
parser.add_argument("-v", "--verbose", action='store_true', default=False)

# training args
parser.add_argument("-b", "--batch_size", type=int, default=64)

# computing args
parser.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=1)
parser.add_argument("-g", "--gpu", type=int, default=-1)

# MPNN
parser.add_argument("--leaves", action='store_true')
parser.add_argument("-i", "--n_iters", type=int, default=1)

# email
#parser.add_argument("--sender", type=str, default="results74207281@gmail.com")
#parser.add_argument("--password", type=str, default="deeplearning")
#parser.add_argument("--recipient", type=str, default="psn240@nyu.edu")

# debugging
parser.add_argument("--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.silent = not args.verbose

if args.debug:
    args.n_text = 1000
    args.bs = 9
    args.verbose = True
''' CONSTANTS '''
'''----------------------------------------------------------------------- '''

DATA_DIR = '/scratch/xw1435/capstone/data/w-vs-qcd/pickles/'
MODELS_DIR = '/home/xw1435/capstone/dev/jets/'
REPORTS_DIR = '/scratch/xw1435/capstone/data/w-vs-qcd/reports/'

def main():

    #eh = ExperimentHandler(args, REPORTS_DIR)
    #signal_handler = eh.signal_handler

    ''' GET RELATIVE PATHS TO DATA AND MODELS '''
    '''----------------------------------------------------------------------- '''
    with open(args.model_list_filename, "r") as f:
        model_paths = [l.strip('\n') for l in f.readlines() if l[0] != '#']

    with open(args.data_list_filename, "r") as f:
        data_paths = [l.strip('\n') for l in f.readlines() if l[0] != '#']

    logging.info("DATA PATHS\n{}".format("\n".join(data_paths)))
    logging.info("MODEL PATHS\n{}".format("\n".join(model_paths)))


    ''' Build Distribution '''
    '''----------------------------------------------------------------------- '''
    
    for data_path in data_paths:

        logging.info('Building ROCs for models trained on {}'.format(data_path))
        tf = load_tf(DATA_DIR, "{}-train.pickle".format(data_path))
        if args.set == 'test':
            data = load_test(tf, DATA_DIR, "{}-test.pickle".format(data_path), args.n_test)
        elif args.set == 'valid':
            data = load_test(tf, DATA_DIR, "{}-valid.pickle".format(data_path), args.n_test)
        elif args.set == 'train':
            data = load_test(tf, DATA_DIR, "{}-train.pickle".format(data_path), args.n_test)

        for model_path in model_paths:
            logging.info('\tBuilding Distribution for instances of {}'.format(model_path))
            y,y_pred = get_prediction(data, os.path.join(MODELS_DIR, model_path), args.batch_size)
    
    ''' PLOT Distribution '''
    '''----------------------------------------------------------------------- '''
    plt.hist(y_pred[y==0], bins=100, normed=1, histtype="step", label="$p(f(X)=0)$")
    plt.hist(y_pred[y==1], bins=100, normed=1, histtype="step", label="$p(f(X)=1)$")
    
    plt.legend(loc="best")
    #plt.ylim(0,4)
    plt.xlabel("$f(X)$")
    plt.ylabel("$p(f(X))$")
    plt.grid()
    plt.legend(loc="upper left")
    plt.savefig("distribution.pdf")
    plt.show()
    

if __name__ == '__main__':
    main()
