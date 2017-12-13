import torch
from torch.optim import Adam, lr_scheduler
import copy
import numpy as np
import logging
import pickle
import time
import os
import argparse
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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
from architectures import AdversarialParticleEmbedding

from loading import load_data
from loading import load_tf
from loading import crop

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

# data args
parser.add_argument("-f", "--filename", type=str, default='antikt-kt')
parser.add_argument("--data_dir", type=str, default=DATA_DIR)
parser.add_argument("-n", "--n_train", type=int, default=-1)
parser.add_argument("--n_valid", type=int, default=27000)
parser.add_argument("--dont_add_cropped", action='store_true', default=False)
parser.add_argument("-p", "--pileup", action='store_true', default=False)

# general model args
parser.add_argument("-m", "--model_type", help="index of the model you want to train - look in constants.py for the model list", type=int, default=0)
parser.add_argument("--features", type=int, default=7)
parser.add_argument("--hidden", type=int, default=40)

# logging args
parser.add_argument("-s", "--silent", action='store_true', default=False)
parser.add_argument("-v", "--verbose", action='store_true', default=False)
parser.add_argument("--extra_tag", type=int, default=0)

# loading previous models args
parser.add_argument("-l", "--load", help="model directory from which we load a state_dict", type=str, default=None)
parser.add_argument("-r", "--restart", help="restart a loaded model from where it left off", action='store_true', default=False)

# training args
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-b", "--batch_size", type=int, default=100)
parser.add_argument("-a", "--step_size", type=float, default=0.001)
parser.add_argument("-d", "--decay", type=float, default=.94)

# computing args
parser.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=None)
parser.add_argument("-g", "--gpu", type=str, default="")

# MPNN
parser.add_argument("--not_leaves", action='store_true')
parser.add_argument("-i", "--iters", type=int, default=0)

# email
parser.add_argument("--sender", type=str, default="results74207281@gmail.com")
parser.add_argument("--password", type=str, default="deeplearning")

# debugging
parser.add_argument("--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)

#Advisory loss setup
parser.add_argument("--lmbda", type=float, default=1.0)


args = parser.parse_args()

if args.debug:
    args.hidden = 1
    args.batch_size = 9
    args.verbose = True
    args.epochs = 3
    args.n_train = 1000
    args.seed = 1
    args.iters = 1

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if args.n_train <= 5 * args.n_valid and args.n_train > 0:
    args.n_valid = args.n_train // 5
args.recipient = RECIPIENT
args.leaves = not args.not_leaves
if args.pileup:
    args.filename = 'antikt-kt-pileup40'
def train(args):
    _, Transform, model_type = TRANSFORMS[args.model_type]
    args.root_exp_dir = os.path.join(MODELS_DIR,model_type, str(args.iters))

    eh = ExperimentHandler(args)

    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    logging.warning("Loading pileup antikt-kt-pileup40 data...")

    tf_pileup_40 = load_tf(args.data_dir, "{}-train.pickle".format(args.filename))
    X_pileup_40, y_pileup_40 = load_data(args.data_dir, "{}-train.pickle".format(args.filename))
    for ij, jet in enumerate(X_pileup_40):
        jet["content"] = tf_pileup_40.transform(jet["content"])

    if args.n_train > 0:
        indices = torch.randperm(len(X_pileup_40)).numpy()[:args.n_train]
        X_pileup_40 = [X_pileup_40[i] for i in indices]
        y_pileup_40 = y_pileup_40[indices]

    logging.warning("Splitting into train and validation...")

    X_train_pileup_40, X_valid_uncropped_pileup_40, y_train_pileup_40, y_valid_uncropped_pileup_40 = train_test_split(X_pileup_40, y_pileup_40, test_size=args.n_valid, random_state=0)
    logging.warning("\traw train size = %d" % len(X_train_pileup_40))
    logging.warning("\traw valid size = %d" % len(X_valid_uncropped_pileup_40))

    X_valid_pileup_40, y_valid_pileup_40, cropped_indices_40, w_valid_40 = crop(X_valid_uncropped_pileup_40, y_valid_uncropped_pileup_40, pileup_lvl=40, return_cropped_indices=True, pileup=args.pileup)
    # add cropped indices to training data
    if not args.dont_add_cropped:
        X_train_pileup_40.extend([x for i, x in enumerate(X_valid_uncropped_pileup_40) if i in cropped_indices_40])
        y_train_pileup_40 = [y for y in y_train_pileup_40]
        y_train_pileup_40.extend([y for i, y in enumerate(y_valid_uncropped_pileup_40) if i in cropped_indices_40])
        y_train_pileup_40 = np.array(y_train_pileup_40)
    
    Z_train_pileup_40 = [0]*len(y_train_pileup_40)
    Z_valid_pileup_40 = [0]*len(y_valid_pileup_40)
    
    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    logging.warning("Loading pileup antikt-kt-pileup50 data...")
    args.filename = 'antikt-kt-pileup50'
    args.pileup = False
    
    tf_pileup_50 = load_tf(args.data_dir, "{}-train.pickle".format(args.filename))
    X_pileup_50, y_pileup_50 = load_data(args.data_dir, "{}-train.pickle".format(args.filename))
    for ij, jet in enumerate(X_pileup_50):
        jet["content"] = tf_pileup_50.transform(jet["content"])

    if args.n_train > 0:
        indices = torch.randperm(len(X_pileup_50)).numpy()[:args.n_train]
        X_pileup_50 = [X_pileup_50[i] for i in indices]
        y_pileup_50 = y_pileup_50[indices]

    logging.warning("Splitting into train and validation...")

    X_train_pileup_50, X_valid_uncropped_pileup_50, y_train_pileup_50, y_valid_uncropped_pileup_50 = train_test_split(X_pileup_50, y_pileup_50, test_size=args.n_valid, random_state=0)
    logging.warning("\traw train size = %d" % len(X_train_pileup_50))
    logging.warning("\traw valid size = %d" % len(X_valid_uncropped_pileup_50))

    X_valid_pileup_50, y_valid_pileup_50, cropped_indices_50, w_valid_50 = crop(X_valid_uncropped_pileup_50, y_valid_uncropped_pileup_50, pileup_lvl=50, return_cropped_indices=True, pileup=args.pileup)
    # add cropped indices to training data
    if not args.dont_add_cropped:
        X_train_pileup_50.extend([x for i, x in enumerate(X_valid_uncropped_pileup_50) if i in cropped_indices_50])
        y_train_pileup_50 = [y for y in y_train_pileup_50]
        y_train_pileup_50.extend([y for i, y in enumerate(y_valid_uncropped_pileup_50) if i in cropped_indices_50])
        y_train_pileup_50 = np.array(y_train_pileup_50)
    
    Z_train_pileup_50 = [1]*len(y_train_pileup_50)
    Z_valid_pileup_50 = [1]*len(y_valid_pileup_50)
    
    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    logging.warning("Loading pileup antikt-kt-pileup60 data...")
    args.filename = 'antikt-kt-pileup60'
    args.pileup = False
    
    tf_pileup_60 = load_tf(args.data_dir, "{}-train.pickle".format(args.filename))
    X_pileup_60, y_pileup_60 = load_data(args.data_dir, "{}-train.pickle".format(args.filename))
    for ij, jet in enumerate(X_pileup_60):
        jet["content"] = tf_pileup_60.transform(jet["content"])

    if args.n_train > 0:
        indices = torch.randperm(len(X_pileup_60)).numpy()[:args.n_train]
        X_pileup_60 = [X_pileup_60[i] for i in indices]
        y_pileup_60 = y_pileup_60[indices]

    logging.warning("Splitting into train and validation...")

    X_train_pileup_60, X_valid_uncropped_pileup_60, y_train_pileup_60, y_valid_uncropped_pileup_60 = train_test_split(X_pileup_60, y_pileup_60, test_size=args.n_valid, random_state=0)
    logging.warning("\traw train size = %d" % len(X_train_pileup_60))
    logging.warning("\traw valid size = %d" % len(X_valid_uncropped_pileup_60))

    X_valid_pileup_60, y_valid_pileup_60, cropped_indices_60, w_valid_60 = crop(X_valid_uncropped_pileup_60, y_valid_uncropped_pileup_60, pileup_lvl=60, return_cropped_indices=True, pileup=args.pileup)
    # add cropped indices to training data
    if not args.dont_add_cropped:
        X_train_pileup_60.extend([x for i, x in enumerate(X_valid_uncropped_pileup_60) if i in cropped_indices_60])
        y_train_pileup_60 = [y for y in y_train_pileup_60]
        y_train_pileup_60.extend([y for i, y in enumerate(y_valid_uncropped_pileup_60) if i in cropped_indices_60])
        y_train_pileup_60 = np.array(y_train_pileup_60)
    
    Z_train_pileup_60 = [2]*len(y_train_pileup_60)
    Z_valid_pileup_60 = [2]*len(y_valid_pileup_60)
    
    X_train = np.concatenate((X_train_pileup_40, X_train_pileup_50, X_train_pileup_60), axis=0)
    X_valid = np.concatenate((X_valid_pileup_40, X_valid_pileup_50, X_valid_pileup_60), axis=0)
    y_train = np.concatenate((y_train_pileup_40, y_train_pileup_50, y_train_pileup_60), axis=0)
    y_valid = np.concatenate((y_valid_pileup_40, y_valid_pileup_50, y_valid_pileup_60), axis=0)
    Z_train = np.concatenate((Z_train_pileup_40, Z_train_pileup_50, Z_train_pileup_60), axis=0)
    Z_valid = np.concatenate((Z_valid_pileup_40, Z_valid_pileup_50, Z_valid_pileup_60), axis=0)
    w_valid = np.concatenate((w_valid_40, w_valid_50, w_valid_60), axis=0)
    
    X_train, y_train, Z_train = shuffle(X_train, y_train, Z_train, random_state=0)
    X_valid, y_valid, Z_valid = shuffle(X_valid, y_valid, Z_valid, random_state=0)
    
    logging.warning("\tfinal X train size = %d" % len(X_train))
    logging.warning("\tfinal X valid size = %d" % len(X_valid))
    logging.warning("\tfinal Y train size = %d" % len(y_train))
    logging.warning("\tfinal Y valid size = %d" % len(y_valid))
    logging.warning("\tfinal Z train size = %d" % len(Z_train))
    logging.warning("\tfinal Z valid size = %d" % len(Z_valid))
    logging.warning("\tfinal w valid size = %d" % len(w_valid))
    

    ''' MODEL '''
    '''----------------------------------------------------------------------- '''
    # Initialization
    logging.info("Initializing model...")
    Predict = PredictFromParticleEmbedding
    if args.load is None:
        model_kwargs = {
            'features': args.features,
            'hidden': args.hidden,
            'iters': args.iters,
            'leaves': args.leaves,
            'batch' : args.batch_size,
        }
        logging.info('No previous models')
        model = Predict(Transform, **model_kwargs)
        adversarial_model = AdversarialParticleEmbedding(**model_kwargs)
        settings = {
            "transform": Transform,
            "predict": Predict,
            "model_kwargs": model_kwargs,
            "step_size": args.step_size,
            "args": args,
            }
    else:
        with open(os.path.join(args.load, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f, encoding='latin-1')
            Transform = settings["transform"]
            Predict = settings["predict"]
            model_kwargs = settings["model_kwargs"]

        model = PredictFromParticleEmbedding(Transform, **model_kwargs)

        try:
            with open(os.path.join(args.load, 'cpu_model_state_dict.pt'), 'rb') as f:
                state_dict = torch.load(f)
        except FileNotFoundError as e:
            with open(os.path.join(args.load, 'model_state_dict.pt'), 'rb') as f:
                state_dict = torch.load(f)

        model.load_state_dict(state_dict)

        if args.restart:
            args.step_size = settings["step_size"]

    logging.warning(model)
    logging.warning(adversarial_model)
    
    out_str = 'Number of parameters: {}'.format(sum(np.prod(p.data.numpy().shape) for p in model.parameters()))
    out_str_adversarial = 'Number of parameters: {}'.format(sum(np.prod(p.data.numpy().shape) for p in adversarial_model.parameters()))
    logging.warning(out_str)
    logging.warning(out_str_adversarial)
    
    if torch.cuda.is_available():
        logging.warning("Moving model to GPU")
        model.cuda()
        logging.warning("Moved model to GPU")
    else:
        logging.warning("No cuda")

    eh.signal_handler.set_model(model)
    
    ''' OPTIMIZER AND LOSS '''
    '''----------------------------------------------------------------------- '''
    
    logging.info("Building optimizer...")
    optimizer = Adam(model.parameters(), lr=args.step_size)
    optimizer_adv = Adam(adversarial_model.parameters(), lr=args.step_size)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)
    scheduler_adv = lr_scheduler.ExponentialLR(optimizer_adv, gamma=args.decay)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    n_batches = int(len(X_train) // args.batch_size)
    best_score = [-np.inf]  # yuck, but works
    best_model_state_dict = copy.deepcopy(model.state_dict())
    
    def loss_adversarial(y_pred, y):
        return -(y * torch.log(y_pred) + (1. - y) * torch.log(1. - y_pred))
    
    def loss(y_pred, y):
        l = log_loss(y, y_pred.squeeze(1)).mean()
        return l

    ''' VALIDATION '''
    '''----------------------------------------------------------------------- '''
    
    def callback(epoch, iteration, model):

        if iteration % n_batches == 0:
            t0 = time.time()
            model.eval()

            offset = 0; train_loss = []; valid_loss = []
            yy, yy_pred = [], []
            for i in range(len(X_valid) // args.batch_size):
                idx = slice(offset, offset+args.batch_size)
                Xt, yt = X_train[idx], y_train[idx]
                X_var = wrap_X(Xt); y_var = wrap(yt)
                y_pred_1 = model(X_var)
                tl = unwrap(loss(y_pred_1, y_var)); train_loss.append(tl)
                X = unwrap_X(X_var); y = unwrap(y_var)

                Xv, yv = X_valid[idx], y_valid[idx]
                X_var = wrap_X(Xv); y_var = wrap(yv)
                y_pred = model(X_var)
                vl = unwrap(loss(y_pred, y_var)); valid_loss.append(vl)
                Xv = unwrap_X(X_var); yv = unwrap(y_var); y_pred = unwrap(y_pred)
                yy.append(yv); yy_pred.append(y_pred)

                offset+=args.batch_size

            train_loss = np.mean(np.array(train_loss))
            valid_loss = np.mean(np.array(valid_loss))
            yy = np.concatenate(yy, 0)
            yy_pred = np.concatenate(yy_pred, 0)

            t1=time.time()
            logging.info("Modeling validation data took {}s".format(t1-t0))
            logging.info(len(yy_pred))
            logging.info(len(yy))
            logging.info(len(w_valid))
            logdict = dict(
                epoch=epoch,
                iteration=iteration,
                yy=yy,
                yy_pred=yy_pred,
                w_valid=w_valid[:len(yy_pred)],
                #w_valid=w_valid,
                train_loss=train_loss,
                valid_loss=valid_loss,
                settings=settings,
                model=model
            )
            eh.log(**logdict)

            scheduler.step(valid_loss)
            model.train()
    
    ''' TRAINING '''
    '''----------------------------------------------------------------------- '''
    eh.save(model, settings)
    logging.warning("Training...")
    iteration=1
    loss_rnn = []
    loss_adv = []
    loss = []
    logging.info("Lambda selected = %.8f" % args.lmbda)
    for i in range(args.epochs):
        logging.info("epoch = %d" % i)
        logging.info("step_size = %.8f" % settings['step_size'])
        t0 = time.time()
        for _ in range(n_batches):
            iteration += 1
            model.train()
            adversarial_model.train()
            optimizer.zero_grad()
            optimizer_adv.zero_grad()
            start = torch.round(torch.rand(1) * (len(X_train) - args.batch_size)).numpy()[0].astype(np.int32)
            idx = slice(start, start+args.batch_size)
            X, y, Z = X_train[idx], y_train[idx], Z_train[idx]
            X_var = wrap_X(X); y_var = wrap(y); 
            #Z_var = wrap(Z, 'long')
            y_pred = model(X_var)
            #l = loss(y_pred, y_var) - loss(adversarial_model(y_pred), Z_var)
            #print(adversarial_model(y_pred))
            Z_var = Variable(torch.squeeze(torch.from_numpy(Z)))
            #print(Z_var)
            l_rnn = loss(y_pred, y_var)
            loss_rnn.append(l_rnn.data.cpu().numpy()[0])
            l_adv = F.nll_loss(adversarial_model(y_pred), Z_var)
            loss_adv.append(l_adv.data.cpu().numpy()[0])
            l = l_rnn - (args.lmbda*l_adv)
            loss.append(l.data.cpu().numpy()[0])
            #Taking step on classifier
            optimizer.step()
            l.backward(retain_graph=True)
            
            #Taking step on advesarial
            optimizer_adv.step()
            l_adv.backward()
            
            X = unwrap_X(X_var); y = unwrap(y_var)
            callback(i, iteration, model)
        t1 = time.time()
        logging.info("Epoch took {} seconds".format(t1-t0))

        scheduler.step()
        scheduler_adv.step()
        settings['step_size'] = args.step_size * (args.decay) ** (i + 1)
    #logging.info(loss_rnn)
    #logging.info('==================================================')
    #logging.info(loss_adv)
    logging.info('PID : %d' % os.getpid())
    pathset = os.path.join(args.data_dir, str(os.getpid()))
    os.mkdir(pathset)
    np.save(os.path.join(pathset, 'rnn_loss.csv'), np.array(loss_rnn))
    np.save(os.path.join(pathset, 'adv_loss.csv'), np.array(loss_adv))
    eh.finished()
    


if __name__ == "__main__":
    train(args)