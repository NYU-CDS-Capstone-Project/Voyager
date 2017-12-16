import torch
from torch.optim import Adam#, lr_scheduler
import copy
import numpy as np
import logging
import pickle
import time
import os
import argparse
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from data_ops.wrapping import wrap
from data_ops.wrapping import unwrap
from data_ops.wrapping import wrap_X
from data_ops.wrapping import unwrap_X

from misc.constants import *
from misc.handlers import ExperimentHandler
from misc.loggers import StatsLogger

from monitors.losses import *
from monitors.monitors import *

from architectures import PredictFromParticleEmbedding
#from architectures import AdversarialParticleEmbedding

from loading import load_data
from loading import load_tf
from loading import crop

from sklearn.utils import shuffle

filename = 'antikt-kt'
data_dir = '/scratch/psn240/capstone/data/w-vs-qcd/pickles/'
tf = load_tf(data_dir, "{}-train.pickle".format(filename))
X, y = load_data(data_dir, "{}-train.pickle".format(filename))
for ij, jet in enumerate(X):
    jet["content"] = tf.transform(jet["content"])
Z = [0]*len(y)

print(len(X))
print(len(y))

filename = 'antikt-kt-pileup25-new'
data_dir = '/scratch/psn240/capstone/data/w-vs-qcd/pickles/'
tf_pileup = load_tf(data_dir, "{}-train.pickle".format(filename))
X_pileup, y_pileup = load_data(data_dir, "{}-train.pickle".format(filename))
for ij, jet in enumerate(X_pileup):
    jet["content"] = tf_pileup.transform(jet["content"])
Z_pileup = [1]*len(y)

print(len(X_pileup))
print(len(y_pileup))

X_combined = np.concatenate((X, X_pileup), axis=0)
y_combined = np.concatenate((y, y_pileup), axis=0)
Z_combined = np.concatenate((Z, Z_pileup), axis=0)
print(len(X_combined))
print(len(y_combined))
print(len(Z_combined))

print(Z_combined[0:100])
X_combined, y_combined, Z_combined = shuffle(X_combined, y_combined, Z_combined, random_state=0)
print(Z_combined[0:100])
