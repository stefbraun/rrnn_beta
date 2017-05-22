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
from ctc_utils import *


def train(train_path, test_path, drop_p, inp_dims, rnn_size, out_size, GRAD_CLIP, seed, lr, max_frame_size,
          normalization, shuffle_type, enable_gauss, int_devil, thresh_devil, thresh, mask_devil):
    # Create symbolic vars
    input_var = T.ftensor3('my_input_var')
    input_var_lens = T.ivector('my_input_var_lens')
    mask_var = T.matrix('my_mask')
    target_var = T.ivector('my_targets')
    target_var_lens = T.ivector('my_targets_lens')

    # Random seed
    np.random.RandomState(seed)
    new_rng = np.random.RandomState(seed)
    lasagne.random.set_rng(new_rng)
    lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(seed)
    random.seed(seed)

    # Get network
    network = get_ctc_net(input_var, mask_var, inp_dims, rnn_size, out_size, GRAD_CLIP, drop_p)

    # Compile
    train_fn, val_fn, pred_fn = get_train_and_val_fn_ctc(input_var, input_var_lens, mask_var, target_var,
                                                         target_var_lens, network, lr)

    # Training
    ERRS=[]
    for ep in range(1, 200):
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

            loss = train_fn(bX, b_lenX, maskX, bY, b_lenY)
            train_loss += loss

        dev_it = BatchIterator()
        dev_loss = 0
        all_guessed_labels = []
        all_target_labels = []
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
            loss = val_fn(bX, b_lenX, maskX, bY, b_lenY)
            dev_loss += loss

            pred = pred_fn(bX, maskX)
            pred_sm = calc_softmax_in_last_dim(pred)

            guessed_labels = convert_prediction_to_transcription(pred_sm, int_to_hr=None,
                                                                 joiner='')  # greedy path, remove repetitions, prepare string
            easier_labels = convert_from_ctc_to_easy_labels(bY, b_lenY)  # ease access to warp-ctc labels
            target_labels = [get_single_decoding(label, int_to_hr=None, joiner='') for label in
                             easier_labels]  # prepare string
            all_guessed_labels.extend(guessed_labels)
            all_target_labels.extend(target_labels)

        PER, WER, CER = calculate_error_rates_dbg(all_target_labels, all_guessed_labels)
        ERRS.append([PER, WER, CER])
        print(PER, WER, CER)
        print(len(all_guessed_labels), all_guessed_labels)
        print(len(all_target_labels), all_target_labels)

        with open('log.csv', 'a') as f:
            c = csv.writer(f)
            if ep == 1:
                c.writerow(
                    ['epoch', 'train_loss', 'dev_loss', 'PER', 'WER', 'CER', 'Timestamp', 'frames', 'padded_frames',
                     'frame_cache',
                     '#train_batches', '#val_batches', '#ukeys train',
                     '#ukeys dev', 'shuffletype', 'Normalization',
                     'dropout', 'enable_gauss', 'int_devil', 'thresh_devil', 'thresh', 'mask_devil', 'best_epoch PER','best_epoch CER', 'seed'])
            c.writerow([ep, train_loss, dev_loss, PER, WER, CER, datetime.now(), train_monitor['frames'],
                        train_monitor['padded_frames'],
                        max_frame_size, train_monitor['batch_no'], dev_monitor['batch_no'],
                        get_uk(train_monitor['epoch_keys']),
                        get_uk(dev_monitor['epoch_keys']), shuffle_type, normalization,
                        drop_p, enable_gauss, int_devil, thresh_devil, thresh, mask_devil, np.argmax(ERRS[-1][0]), np.argmax(ERRS[-1][2]), seed])

    with open('log_super.csv', 'a') as f:
        c = csv.writer(f)
        c.writerow([seed, shuffle_type, normalization,
                    drop_p, enable_gauss, int_devil, thresh_devil, thresh, mask_devil, np.argmax(ERRS[:,0]), np.argmax(ERRS[:,2])])


if __name__ == '__main__':
    # Data
    train_path = '/media/stefbraun/ext4/Dropbox/dataset/tidigits_ctc/train.h5'
    test_path = '/media/stefbraun/ext4/Dropbox/dataset/tidigits_ctc/test.h5'

    # Params
    drop_p = 0.5
    inp_dims = 39
    rnn_size = 500
    out_size = 12
    GRAD_CLIP = 200.
    seeds = range(123, 133)
    lr = 1e-3
    max_frame_size = 25000
    normalization = 'epoch'
    shuffle_type = 'exp'
    enable_gauss = 0

    int_devil = 0
    thresh_devil = 0
    threshs = np.arange(0, 5, 0.5)
    mask_devil = 0

    for thresh in threshs:
        with open('log_super.csv', 'a') as f:
            c = csv.writer(f)

            c.writerow(['seed', 'shuffletype', 'Normalization',
                        'dropout', 'enable_gauss', 'int_devil', 'thresh_devil', 'thresh', 'mask_devil', 'PERarg', 'CERarg'])
        for seed in seeds:
            train(train_path=train_path, test_path=test_path, drop_p=drop_p, inp_dims=inp_dims,
                  rnn_size=rnn_size,
                  out_size=out_size, GRAD_CLIP=GRAD_CLIP, seed=seed, lr=lr,
                  max_frame_size=max_frame_size,
                  normalization=normalization, shuffle_type=shuffle_type,
                  enable_gauss=enable_gauss, int_devil=int_devil,
                  thresh_devil=thresh_devil, thresh=thresh, mask_devil=mask_devil)
