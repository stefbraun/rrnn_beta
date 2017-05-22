from lasagne_lib import *
import theano.tensor as T
from pem_lib import BatchIterator
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from pem_lib import BatchIterator, get_uk
from ep_lib_python import SimpleEpochIterator


def train(train_path, test_path, drop_p, inp_dims, rnn_size, out_size, GRAD_CLIP, seed, lr, max_frame_size,
          normalization, shuffle_type, enable_gauss, int_devil, thresh_devil, thresh, mask_devil):
    # Create symbolic vars
    input_var = T.ftensor3('my_input_var')
    mask_var = T.matrix('my_mask')
    target_var = T.ivector('my_targets')

    # Random seed
    # np.random.RandomState(seed)
    # new_rng = np.random.RandomState(seed)
    # lasagne.random.set_rng(new_rng)
    # lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(seed)
    # random.seed(seed)

    # Get network
    network = get_rrnn_net(input_var, mask_var, inp_dims, rnn_size, out_size, GRAD_CLIP, drop_p)

    print('# network parameters: ' + str(lasagne.layers.count_params(network)))

    # Compile
    train_fn, val_fn, pred_fn = get_train_and_val_fn(input_var, mask_var, target_var, network, lr)

    # Training

    ep_iterator = SimpleEpochIterator()

    dev_acc = []
    train_acc = []
    for ep, train_path, dev_path, ep_monitor, enable_gauss in ep_iterator.flow(start_epoch=1,
                                                                                  num_epochs=250,
                                                                                  max_patience=20,
                                                                                  ep_type='baseline',
                                                                                  train_dataset=train_path,
                                                                                  dev_dataset=test_path, debug=0):
        train_preds = []
        train_targets = []
        train_it = BatchIterator()
        train_loss = 0
        for bX, b_lenX, maskX, bY, b_lenY, train_monitor in train_it.flow(epoch=ep, h5=train_path,
                                                                          shuffle_type=shuffle_type,
                                                                          max_frame_size=max_frame_size,
                                                                          normalization=normalization,
                                                                          enable_gauss=enable_gauss):
            if int_devil == 1:
                bX = bX.astype(dtype='int32')
                bX = bX.astype(dtype='float32')
            if thresh_devil == 1:
                indices = np.abs(bX) < thresh
                bX[indices] = 0
            if mask_devil == 1:
                maskX = np.ones(maskX.shape, dtype='float32')

            loss = train_fn(bX, maskX, bY)
            train_loss += loss

            pred = pred_fn(bX, maskX)
            train_preds.extend(pred.argmax(axis=1).tolist())
            train_targets.extend(bY.tolist())
        train_acc.append(100.0 * accuracy_score(train_targets, train_preds))

        dev_preds = []
        dev_targets = []
        dev_it = BatchIterator()
        dev_loss = 0
        for bX, b_lenX, maskX, bY, b_lenY, dev_monitor in dev_it.flow(epoch=1, h5=test_path,
                                                                      shuffle_type=shuffle_type,
                                                                      max_frame_size=max_frame_size,
                                                                      normalization=normalization,
                                                                      enable_gauss=enable_gauss):
            if int_devil == 1:
                bX = bX.astype(dtype='int32')
                bX = bX.astype(dtype='float32')
            if thresh_devil == 1:
                indices = np.abs(bX) < thresh
                bX[indices] = 0
            if mask_devil == 1:
                maskX = np.ones(maskX.shape, dtype='float32')
            loss = val_fn(bX, maskX, bY)
            dev_loss += loss
            pred = pred_fn(bX, maskX)
            dev_preds.extend(pred.argmax(axis=1).tolist())
            dev_targets.extend(bY.tolist())

        dev_acc.append(100.0 * accuracy_score(dev_targets, dev_preds))
        ep_iterator.mon_var.append(dev_acc[-1])

        print('{} {} {} {:5.2f} {:5.2f} {:5.2f}'.format(ep, ep_monitor['patience'],bX.shape, train_loss, train_acc[-1], dev_acc[-1]))

        with open('log.csv', 'a') as f:
            c = csv.writer(f)
            if ep == 1:
                c.writerow(
                    ['epoch', 'train_loss', 'dev_loss', 'train_acc', 'dev_acc', 'Timestamp', 'frames', 'padded_frames',
                     'frame_cache',
                     '#train_batches', '#val_batches', '#ukeys train',
                     '#ukeys dev', 'shuffletype', 'Normalization',
                     'dropout', 'enable_gauss', 'int_devil', 'thresh_devil', 'thresh', 'mask_devil', 'targ', 'tacc',
                     'darg', 'dacc', 'seed', 'lr', 'GRAD_CLIP', 'rnn_size'])
            c.writerow([ep, train_loss, dev_loss, train_acc[-1], dev_acc[-1], datetime.now(), train_monitor['frames'],
                        train_monitor['padded_frames'],
                        max_frame_size, train_monitor['batch_no'], dev_monitor['batch_no'],
                        get_uk(train_monitor['epoch_keys']),
                        get_uk(dev_monitor['epoch_keys']), shuffle_type, normalization,
                        drop_p, enable_gauss, int_devil, thresh_devil, thresh, mask_devil, np.argmax(train_acc),
                        np.max(train_acc), np.argmax(dev_acc), np.max(dev_acc), seed, lr, GRAD_CLIP, rnn_size])

    with open('log_super.csv', 'a') as f:
        c = csv.writer(f)
        c.writerow([seed, rnn_size, GRAD_CLIP, lr,
                    drop_p, enable_gauss,shuffle_type, normalization, np.argmax(train_acc),
                    np.max(train_acc), np.argmax(dev_acc), np.max(dev_acc)])


if __name__ == '__main__':
    # Data
    train_path = '/media/stefbraun/ext4/Dropbox/dataset/tidigits_iso/train.h5'
    test_path = '/media/stefbraun/ext4/Dropbox/dataset/tidigits_iso/test.h5'

    # Params to sample
    drop_p = [0, 0.25, 0.5]
    inp_dims = 39
    rnn_size = [200, 300]
    out_size = 11
    GRAD_CLIP = [20., 100., 200.]
    seeds = range(123, 134)
    lr = [1e-2, 1e-3, 1e-4]

    max_frame_size = 100000
    normalization = 'epoch'
    shuffle_type = 'exp'
    enable_gauss = 0

    int_devil = 0
    thresh_devil = 0
    threshs = np.arange(0, 20, 0.5)
    mask_devil = 0

    with open('log_super.csv', 'a') as f:
        c = csv.writer(f)

        c.writerow(['seed', 'rnn_size', 'GRAD_CLIP', 'lr',
                    'dropout', 'enable_gauss', 'shuffletype', 'Normalization','targ', 'tacc',
                    'darg',
                    'dacc'])

    for i in range(0, 200):
        train(train_path=train_path, test_path=test_path, drop_p=random.choice(drop_p), inp_dims=inp_dims,
              rnn_size=random.choice(rnn_size),
              out_size=out_size, GRAD_CLIP=random.choice(GRAD_CLIP), seed=random.choice(seeds), lr=random.choice(lr),
              max_frame_size=max_frame_size,
              normalization=normalization, shuffle_type=shuffle_type,
              enable_gauss=enable_gauss, int_devil=int_devil,
              thresh_devil=thresh_devil, thresh=0, mask_devil=mask_devil)
