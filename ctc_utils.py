import theano.tensor as T
import numpy as np
import lasagne
import os
import cPickle as pickle
import matplotlib.pyplot as plt
import theano
import time
import editdistance
import csv


def save_model(filename, suffix, model, log=None, announce=True):
    # Build filename
    filename = '{}_{}'.format(filename, suffix)
    # Acquire Data
    data = lasagne.layers.get_all_param_values(model)
    # Store in separate directory
    filename = os.path.join('./models/', filename)
    # Inform user
    if announce:
        print('Saving to: {}'.format(filename))
    # Generate parameter filename and dump
    param_filename = '%s.params' % (filename)
    with open(param_filename, 'w') as f:
        pickle.dump(data, f)
    # Generate log filename and dump
    if log is not None:
        log_filename = '%s.log' % (filename)
        with open(log_filename, 'w') as f:
            pickle.dump(log, f)


def load_model(filename, model):
    # Build filename
    filename = os.path.join('./models/', '%s.params' % (filename))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)
    return model


def load_log(filename):
    filename = os.path.join('./models/', '%s.log' % (filename))
    with open(filename, 'r') as f:
        log = pickle.load(f)
    return log


def store_in_log(log, kv_pairs):
    # Quick helper function to append values to keys in a log
    for k, v in kv_pairs.items():
        log[k].append(v)
    return log


def replace_nans_with_zero(updates):
    # Replace all nans with zeros
    for k, v in updates.items():
        k = T.switch(T.eq(v, np.nan), float(0.), v)
    print('Warning: replaced nans')
    return updates


def get_net_output_fn(fn_inputs, network):
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    out_fn = theano.function(fn_inputs, test_prediction)
    return out_fn


def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


def calc_softmax_in_last_dim(result):
    if len(result.shape) != 3:
        raise AssertionError('Result should be a [A x B x feats] tensor.')
    for outer_idx in range(result.shape[0]):
        for inner_idx in range(result.shape[1]):
            result[outer_idx, inner_idx, :] = softmax(result[outer_idx, inner_idx, :])
    return result


def eliminate_duplicates_and_blanks(guess_vec):
    """
    Map sequences of frame-level CTC labels to single lexicon units

    :param guess_vec: 1d vector with guessed label for each time frame
    :return: guess_vec with duplicates and blanks eliminated
    >>> eliminate_duplicates_and_blanks(guess_vec=[0,0,1,2,2,3,1,0,3,3])
    [1, 2, 3, 1, 3]
    """

    rv = []
    # Remove duplicates
    for item in guess_vec:
        if (len(rv) == 0 or item != rv[-1]):
            rv.append(item)
    # Remove blanks (= labels with index 0 by convention)
    final_rv = []
    for item in rv:
        if item != 0:
            final_rv.append(item)
    return final_rv


def plot_err_from_log(log, prefix='b_'):
    plt.semilogy(log[prefix + 'val_err'])
    plt.hold(True)
    plt.semilogy(log[prefix + 'train_err'])
    plt.grid(True, which='both')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend(['Validation', 'Training'])


def get_single_decoding(guess_vec, int_to_hr=None, joiner=''):
    """
    Join a label sequence with separators '-' for integers or '' for characters

    :param guess_vec: 1d label vector
    :param int_to_hr: 1d vector, each index is filled by a string
    :param joiner: string between labels
    :return: 1d vector with guessed label

    >>> get_single_decoding(guess_vec=[2,4,1,0])
    '2-4-1-0'
    """
    if int_to_hr is None:
        guessed_label = '-'.join([str(item) for item in guess_vec])
    else:
        guessed_label = joiner.join([int_to_hr[item] for item in guess_vec if item != 0])
    return guessed_label


def get_single_max_decoding(result, int_to_hr=None, joiner=''):
    """Return the greedy path throug a single sample and eliminate duplicates and blanks

    result : Time X Features 2d Matrix of label probabilities

    Get greedy decoding of diagonal matrix. Index 0: blank label
    >>> get_single_max_decoding(result=np.diag([1,1,1]), int_to_hr=None, joiner='')
    '1-2'

    """
    guess_vec = np.argmax(result, axis=1)
    guess_vec_elim = eliminate_duplicates_and_blanks(guess_vec)
    return get_single_decoding(guess_vec_elim, int_to_hr, joiner)


def plot_ctc_predictions_line(result, label, int_to_hr=None, joiner='', show_legend=True):
    # Transpose the result and squeeze it down to 2D if it isn't already
    sqz_result = np.squeeze(result).T
    # Plot all the lines, store the index
    legend_list = []
    for idx, result_line in enumerate(sqz_result):
        plt.plot(result_line)
        if idx == 0 or int_to_hr is None:
            legend_list.append(str(idx))
        else:
            legend_list.append(int_to_hr[idx])
    # Put on a legend
    if show_legend:
        plt.legend(legend_list, loc=5)
    # Label it with the guess and the predictions
    guessed_label = get_single_max_decoding(result, int_to_hr, joiner)
    plt.title('Label: {} \n Guessed: {}'.format(label, guessed_label))
    # Formatting
    plt.grid(True, which='both')
    plt.ylim([-0.15, 1.15])


def convert_from_ctc_to_easy_labels(bY, lenY):
    """
    Convert labels in warp_ctc format to nested list of labels

    The returned list is easier to handle for calculation of error rates.

    :param bY: 1d vector with labels flattened over batch
    :param lenY: 1d vector with label length per sample in batch
    :return: nested list of labels
    >>> convert_from_ctc_to_easy_labels(bY=[1,2,3,4,2,4,3],lenY=[3,4])
    [[1, 2, 3], [4, 2, 4, 3]]

    """
    curr_idx = 0
    curr_label = 0
    labels = []
    while curr_idx < len(bY):
        curr_len = lenY[curr_label]
        label_list = bY[curr_idx:curr_idx + curr_len]
        labels.append([item for item in label_list])
        curr_idx += curr_len
        curr_label += 1
    return labels


def make_label_vec(output_mat):
    redundant_label_vec = np.argmax(output_mat, axis=1)
    label_vec = np.array(eliminate_duplicates_and_blanks(redundant_label_vec))
    return label_vec.astype('int32')


def DEPRECATED_calculate_error_rates(target_labels, guessed_labels):
    # Get Phrase Error Rate
    phrases_correct = 0
    for idx, target in enumerate(target_labels):
        if len(target_labels) == len(guessed_labels[idx]) and \
                np.all(target_labels == guessed_labels[idx]):
            phrases_correct += 1
    PER = 1. - (float(phrases_correct) / len(target_labels))
    # Get Word Error Rate
    words_wrong = 0
    total_words = 0
    for lbl_idx, target in enumerate(target_labels):
        guess_words = guessed_labels[lbl_idx].split(' ')
        target_words = target.split(' ')
        max_err = len(target_words)
        errors = int(editdistance.eval(guess_words, target_words))
        words_wrong += np.min((max_err, errors))
        total_words += len(target_words)
    WER = float(words_wrong) / total_words
    # Get Character Error Rate with edit distance
    chars_wrong = 0
    total_chars = 0
    for idx, target in enumerate(target_labels):
        errors = int(editdistance.eval(target, guessed_labels[idx]))
        chars_wrong += np.min((errors, len(target)))
        total_chars += len(target)
    CER = float(chars_wrong) / total_chars
    return PER, WER, CER


def calculate_error_rates_dbg(target_labels, guessed_labels, space_idx=3):
    """
    Calculate phrase error rate, word error rate and character error rate

    Warning: label '3' is considered as <space> label by default.

    Notice:  In early CTC training stages, WER and CER are suspect to be exactly one. This happens because the network
             only outputs blank labels. This blank labels should be removed before being passed to this function. This
             leads to guessed labels that are only lists of empty strings ''. The editdistance between
             an empty string '' and a target string sequence is equal to the length of the target
             string. If then divided by the target string length, the result is 1.


    :param target_labels: 1d vector of strings with characters separated by '-', e.g ['38-1-42','2-44-24']
    :param guessed_labels: 1d vector of strings with characters separated by '-', e.g ['38-1-42','2-44-24']
    :param type: currently only 'int' supported
    :return: phrase error rate, word error rate and character error rate - NOT capped at 100%

    Check for matching case
    >>> calculate_error_rates_dbg(target_labels=['38-1-42-3-37-22','2-44-24'], guessed_labels=['38-1-42-3-37-22','2-44-24'])
    (0.0, 0.0, 0.0)

    Check for non-matching case
    >>> calculate_error_rates_dbg(target_labels=['38-1-42-3-37-22', '2-44', '1'], guessed_labels=['38-1-42-37-22', '2-44-24', '1'])
    (0.6666666666666667, 0.75, 0.2222222222222222)

    Check for kaldis compute-wer 1/2
    >>> calculate_error_rates_dbg(target_labels=['1-3-2-3-33-3-4', '7-3-8-3-9', '5-3-7-3-33','4-3-33-3-2-3-1','1','1'], guessed_labels=['1-3-2-3-33-3-4-3-5', '7-3-8','5-3-8-3-33', '1-3-33-3-5-3-2','1','1-3-2-3-4'])
    (0.8333333333333334, 0.5, 0.46153846153846156)

    Check for kaldis computer-wer 2/2
    >>> calculate_error_rates_dbg(target_labels=['1'], guessed_labels=['1-3-2-3-4'])
    (1.0, 2.0, 4.0)
    """

    # Get Phrase Error Rate Match
    phrases_correct = 0
    for idx, target in enumerate(target_labels):
        if len(target) == len(guessed_labels[idx]) and np.all(target == guessed_labels[idx]):
            phrases_correct += 1
    PER = 1. - (float(phrases_correct) / len(target_labels))

    # Word error rate
    words_wrong = 0
    total_words = 0
    for lbl_idx, target in enumerate(target_labels):
        guess_words = guessed_labels[lbl_idx].split('-' + str(space_idx) + '-')
        target_words = target.split('-' + str(space_idx) + '-')
        errors = int(editdistance.eval(guess_words, target_words))
        words_wrong += errors
        total_words += len(target_words)
    WER = float(words_wrong) / total_words

    # Character error rate
    chars_wrong = 0
    total_chars = 0
    for idx, target in enumerate(target_labels):
        guess_chars = guessed_labels[idx].split('-')
        target_chars = target.split('-')
        errors = int(editdistance.eval(target_chars, guess_chars))
        chars_wrong += errors
        total_chars += len(target_chars)
    CER = float(chars_wrong) / total_chars

    return PER, WER, CER


def get_output_over_dataset(d, out_fn, d_kwargs, int_to_hr, joiner='', announce=True):
    all_guessed_labels = []
    all_target_labels = []

    b_idx = 0
    num_batches = int(np.ceil(float(len(d_kwargs['X'])) / d_kwargs['batch_size']))
    for bX, b_lenX, maskX, bY, b_lenY, s_idxs in d.flow(**d_kwargs):
        start_time = time.time()
        # Get Output
        result = out_fn(bX, maskX)
        # Softmax the result
        result = calc_softmax_in_last_dim(result)
        # Decode guess
        guessed_labels = convert_prediction_to_transcription(result, int_to_hr, joiner)
        # Get correct result
        easier_labels = convert_from_ctc_to_easy_labels(bY, b_lenY)
        # Decode correct result
        labels = [get_single_decoding(label, int_to_hr, joiner) for label in easier_labels]
        # Store results
        all_guessed_labels.extend(guessed_labels)
        all_target_labels.extend(labels)
        # Provide some feedback
        if announce:
            print('Batch {} of {} (FF: {:.2f}%): in {:.2f}ms.'.format(
                b_idx + 1, num_batches,
                np.mean(maskX) * 100.,
                (time.time() - start_time) * 1000.))
        b_idx += 1
    return all_target_labels, all_guessed_labels


def convert_hr_to_int(phrase, hr_to_int_dict, conv_chars=True, allow_oov=False):
    int_list = []
    if conv_chars:
        target = phrase
    else:
        target = phrase.split(' ')
    for hr in target:
        try:
            int_list.append(hr_to_int_dict[hr])
        except KeyError:
            int_list.append(len(hr_to_int_dict.keys()) + 1)  # offset by blank symbol
    return int_list

def convert_int_to_hr(trans_int):
    units_txt = '/media/stefbraun/ext4/audio_group/stefan/lv_snr/units.txt'
    units = []
    with open(units_txt) as f:
        data = f.readlines()
    units = [element.split(' ')[0] for element in data]

    trans_char = []
    for idx in trans_int.split('-'):
        if (int(idx)-1) == 2:
            trans_char.append(' ')
        else:
            trans_char.append(units[int(idx)-1])

    trans_char = ''.join(trans_char)
    return trans_char

def convert_prediction_to_transcription(prediction, int_to_hr, joiner):
    # Prediction : Time X Batch X Features 3D matrix
    prediction = prediction.transpose([1, 0, 2])
    guessed_labels = [get_single_max_decoding(phrase, int_to_hr, joiner) for phrase in prediction]
    return guessed_labels


def correctly_append_y(bY):
    final_y = []
    for item in bY:
        final_y.extend(item)
    return np.array(final_y).astype('int32')
