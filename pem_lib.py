import numpy as np
import h5py
import matplotlib.pyplot as plt
import cPickle as pkl
import warnings
from sklearn.preprocessing import minmax_scale


class BatchIterator(object):
    """
    """

    def flow(self, epoch, h5, max_frame_size=1000, shuffle_type='high_throughput', normalization='epoch',
             enable_gauss=0):
        np.random.seed(epoch)

        with h5py.File(h5) as hf:
            feats_hdd = hf.get('features')
            feats = feats_hdd[:, :]
            feature_lens = np.array(hf.get('feature_lens')).astype(int)
            labels = np.array(hf.get('labels')).astype(int)
            label_lens = np.array(hf.get('label_lens')).astype(int)
            kys = np.array(hf.get('keys'))
            # snr_vals = np.array(hf.get('snr')).astype(int)
            snr_vals = [0, 0]
            feat_idx = idx_to_slice(feature_lens)
            label_idx = idx_to_slice(label_lens)

            # Prepare monitoring
            batch_monitor = {'snr': [np.mean(snr_vals), np.std(snr_vals), np.min(snr_vals), np.max(snr_vals)]}

            # Get some constants
            num_examples = len(feature_lens)
            ndim = len(feats[1, :])

            if normalization == 'epoch':
                ep_mean = np.mean(feats[:, :], axis=0)
                ep_std = np.std(feats[:, :], axis=0)

            if normalization == 'epoch_scale':
                feats = minmax_scale(feats, feature_range=(0, 1), axis=0)

            if shuffle_type == 'none':
                s_idx = range(num_examples)

            elif shuffle_type == 'high_throughput':
                s_idx = np.argsort(feature_lens)

            elif shuffle_type == 'shuffle':
                s_idx = range(num_examples)
                np.random.shuffle(s_idx)

            elif shuffle_type == 'exp':
                s_idx = split_shuffle(split_factor=10, feature_lens=feature_lens, epoch=epoch)

            batches = max_frame_cache(s_idx=s_idx, max_frame_size=max_frame_size, feature_lens=feature_lens)

            if shuffle_type != 'none':
                np.random.shuffle(batches)

            nb_batch = len(batches)
            b = 0

            epoch_keys = []
            frames = 0
            padded_frames = 0

            while b < nb_batch:
                curr_batch = batches[b]

                # Load batch
                batch_feats = []
                batch_labels = []
                batch_keys = []
                for sample in curr_batch:
                    batch_feats.append(feats[slc(sample, feat_idx), :])
                    batch_labels.append(labels[slc(sample, label_idx)])

                    curr_key = kys[sample]
                    # curr_str_key = ''.join(chr(int(i)) for i in curr_key)
                    curr_str_key = str(curr_key)
                    batch_keys.append(curr_str_key)

                # Normalize batch
                if normalization == 'epoch':
                    batch_feats = ep_normalize(batch_feats, ep_mean, ep_std)
                elif (normalization == 'batch') or (normalization == 'sample'):
                    batch_feats = normalize(batch_feats, normalization)

                # Add gaussian noise:
                if enable_gauss == 1:
                    batch_feats = add_gaussian_noise(batch_feats, sigma=0.6)

                # Prepare output
                max_len = np.max(feature_lens[curr_batch])
                bX = np.zeros((len(curr_batch), max_len, ndim), dtype='float32')
                maskX = np.zeros((len(curr_batch), max_len), dtype='float32')

                bY = []
                b_lenY = np.zeros((len(curr_batch)), dtype='int32')
                b_lenX = feature_lens[curr_batch].astype('int32')
                for i, sample in enumerate(batch_feats):
                    len_sample = sample.shape[0]
                    # WARNING change to original after experiment
                    bX[i, :len_sample, :] = sample
                    maskX[i, :len_sample] = np.ones((1, len_sample))

                    # bX[i, max_len-len_sample:, :] = sample
                    # maskX[i, max_len-len_sample:] = np.ones((1, len_sample))
                    # end of warning
                    ctc_labels = np.asarray(batch_labels[i])  # + 1  # shift +1 because 0=blank in CTC training
                    bY.extend(ctc_labels)
                    b_lenY[i] = len(ctc_labels)
                bY = np.asarray(bY, dtype='int32')
                b += 1

                # Get monitoring data
                epoch_keys.append(batch_keys)
                frames += np.sum(b_lenX)
                padded_frames += bX.shape[0] * bX.shape[1]

                batch_monitor['epoch_keys'] = epoch_keys
                batch_monitor['frames'] = frames
                batch_monitor['padded_frames'] = padded_frames
                batch_monitor['s_idx'] = s_idx
                batch_monitor['batch_no'] = b + 1
                yield bX, b_lenX, maskX, bY, b_lenY, batch_monitor


def idx_to_slice(lens):
    idx = []
    lens_cs = np.cumsum(lens)
    for i, len in enumerate(lens):
        idx.append((lens_cs[i] - lens[i], lens_cs[i]))
    return idx


def slc(i, idx):
    return slice(idx[i][0], idx[i][1])


def ep_normalize(batch, ep_mean, ep_std):
    batch_normalized = []
    for sample in batch:
        sample = sample - ep_mean[np.newaxis, :]
        sample = sample / ep_std[np.newaxis, :]
        batch_normalized.append(sample)

    return batch_normalized


def split_shuffle(split_factor, feature_lens, epoch):
    s_idx = np.arange((len(feature_lens)))
    np.random.seed(epoch)
    np.random.shuffle(s_idx)
    s_idx = np.array_split(s_idx, split_factor)

    split_idx = []
    for partial in s_idx:
        partial = partial[np.argsort(feature_lens[partial])]
        split_idx.append(partial)
    split_idx_ar = np.concatenate(split_idx)

    return split_idx_ar


def normalize(batch, normalization):
    """

    :param batch:
    :param normalization:
    :return:
    >>> batch=[np.random.randint(5, size=(200, 123)),np.random.randint(200, size=(200, 123)),np.random.randint(99, size=(200, 123))]
    >>> nb=normalize(batch, 'batch')
    >>> concat=np.concatenate((nb[0], nb[1], nb[2]), axis=0)
    >>> np.allclose(np.mean(concat, axis=0), np.zeros(123))
    True

    >>> np.allclose(np.std(concat, axis=0), np.ones(123))
    True

    >>> nb=normalize(batch, 'sample')
    >>> np.allclose(np.mean(nb[2], axis=0), np.zeros(123))
    True

    >>> np.allclose(np.std(nb[2], axis=0), np.ones(123))
    True
    """
    if normalization == 'batch':
        batch_concat = []
        for sample in batch:
            batch_concat.extend(sample)
        temp = np.asarray(batch_concat)

        b_mean = np.mean(temp, axis=0)
        b_std = np.std(temp, axis=0)

        batch_normalized = []
        for sample in batch:
            sample = sample - b_mean[np.newaxis, :]
            sample = sample / b_std[np.newaxis, :]
            batch_normalized.append(sample)

    elif normalization == 'sample':
        batch_normalized = []
        for sample in batch:
            s_mean = np.mean(sample, axis=0)
            s_std = np.std(sample, axis=0)
            sample = sample - s_mean[np.newaxis, :]
            sample = sample / s_std[np.newaxis, :]

            batch_normalized.append(sample)

    return batch_normalized


def add_gaussian_noise(batch_feats, sigma=0.6):
    """
    :param batch_feats:
    :param sigma:
    :return:

    >>> batch = [np.random.randint(5, size=(200, 123)), np.random.randint(200, size=(200, 123)),np.random.randint(99, size=(200, 123))]
    >>> ng=add_gaussian_noise(batch, sigma=0.6)
    >>> concat_g=np.concatenate((ng[0], ng[1], ng[2]), axis=0)
    >>> concat_b=np.concatenate((batch[0], batch[1], batch[2]), axis=0)
    >>> concat_n=concat_g-concat_b
    >>> np.isclose(0, np.mean(concat_n), atol=1e-2)
    True
    >>> np.isclose(0.6, np.std(concat_n), atol=1e-2)
    True
    """

    mu = 0
    batch_gauss = []
    for sample in batch_feats:
        noise_mat = sigma * np.random.standard_normal(sample.shape)
        sample = sample + noise_mat

        batch_gauss.append(sample)
    return batch_gauss


def check_zmuv(feats, mode):
    err = 0
    mean = np.mean(feats, axis=0)
    zero_mean = np.zeros(feats.shape[1])
    std = np.std(feats, axis=0)
    unit_var = np.ones(feats.shape[1])

    if np.allclose(mean, zero_mean, atol=1e-3) != 1:
        print(mode + ' normalization: zero mean error')
        err = err + 1

    if np.allclose(std, unit_var, atol=1e-3) != 1:
        print(mode + ' normalization: unit variance error')
        err = err + 1

    return err


def max_frame_cache(s_idx, max_frame_size, feature_lens):
    batches = []
    mini_batch = []
    max_len = 0

    for i, sample in enumerate(s_idx):
        # get total frames if new sample was added wrt padding

        max_len = max(feature_lens[sample], max_len)
        total_frames = (len(mini_batch) + 1) * max_len
        # Decide if new sample is added
        if total_frames <= max_frame_size:
            mini_batch.append(sample)
        else:

            # Pycharm Debugging helper variables
            # mbatch_lens=feature_lens[mini_batch]
            # dec = np.max(mbatch_lens)
            # mbatch_length = len(mini_batch)
            # dbg=len(mini_batch)*np.max(feature_lens[mini_batch])

            batches.append(mini_batch)
            mini_batch = [sample]
            max_len = feature_lens[sample]

    batches.append(mini_batch)
    return batches


def get_uk(nested_key_list):
    # currently only 2d lists!!!!
    all_keys = []
    for minibatch in nested_key_list:
        all_keys.extend(minibatch)
    no_unique_keys = len(np.unique(all_keys))
    return no_unique_keys


def dataset_creator(eng, dataset, epoch, SNR, gpu, mode, dbg):
    path_to_WSJ = '/home/stefbraun/data/wsj_wav/'
    path_to_temp = '/media/stefbraun/Data/temp/'

    # path_to_temp = '/home/stefbraun/data/temp/'

    keys_dict = pkl.load(open(path_to_WSJ + 'order.pkl', 'rb'))
    labels_dict = pkl.load(open(path_to_WSJ + 'labels.pkl', 'rb'))
    label_lens_dict = pkl.load(open(path_to_WSJ + 'label_lens.pkl', 'rb'))

    keys = keys_dict[dataset]
    full_keys = [path_to_WSJ + dataset + '/' + dataset + '_' + i for i in keys]

    labels = labels_dict[dataset]
    label_lens = label_lens_dict[dataset]

    if dbg != 0:
        warnings.warn('Dataset creator is in debug mode!')
    if dbg == 2:
        mode = mode + str(dbg) + '#####'
        future = eng.pem_builder_beta(epoch, SNR, full_keys, labels, label_lens, path_to_temp, gpu, mode, async=True)
    elif dbg == 1:
        mode = mode + str(dbg) + '#####'
        full_keys = full_keys[:10]
        labels = labels[:10]
        label_lens = label_lens[:10]
        future = eng.pem_builder_dbg(epoch, SNR, full_keys, labels, label_lens, path_to_temp, gpu, mode, async=True)
    else:
        future = eng.pem_builder_beta(epoch, SNR, full_keys, labels, label_lens, path_to_temp, gpu, mode, async=True)
    return future
