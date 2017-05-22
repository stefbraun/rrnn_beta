import lasagne
import theano
import theano.tensor as T
import numpy as np

from rrnn_beta import RRNNLayer, TimeGate
from plstm_utils import ExponentialUniformInit


def get_gru_net(input_var, mask_var, inp_dim, rnn_size, out_size, GRAD_CLIP, drop_p):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Masking layer
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1 = lasagne.layers.GRULayer(l_in, num_units=rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP,
                                  hid_init=lasagne.init.GlorotUniform())
    h2 = lasagne.layers.DropoutLayer(h1, p=drop_p)
    h3 = lasagne.layers.SliceLayer(h2, -1, axis=1)
    h4 = lasagne.layers.DenseLayer(h3, num_units=rnn_size)
    h5 = lasagne.layers.DropoutLayer(h4, p=drop_p)
    h6 = lasagne.layers.DenseLayer(h5, num_units=out_size, nonlinearity=lasagne.nonlinearities.softmax)

    return h6

def get_rnn_net(input_var, mask_var, inp_dim, rnn_size, out_size, GRAD_CLIP, drop_p):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Masking layer
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1 = lasagne.layers.RecurrentLayer(l_in, num_units=rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP,
                                  hid_init=lasagne.init.GlorotUniform())
    h2 = lasagne.layers.DropoutLayer(h1, p=drop_p)
    h3 = lasagne.layers.SliceLayer(h2, -1, axis=1)
    h4 = lasagne.layers.DenseLayer(h3, num_units=rnn_size)
    h5 = lasagne.layers.DropoutLayer(h4, p=drop_p)
    h6 = lasagne.layers.DenseLayer(h5, num_units=out_size, nonlinearity=lasagne.nonlinearities.softmax)

    return h6

def get_rrnn_net(input_var, mask_var, inp_dim, rnn_size, out_size, GRAD_CLIP, drop_p):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Masking layer
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    # h1 = RRNNLayer(l_in, num_units=rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP)
    h1 = RRNNLayer(l_in,time_input=None,
            num_units=rnn_size,
            mask_input=l_mask,
            nonlinearity=lasagne.nonlinearities.rectify)

    h2 = lasagne.layers.DropoutLayer(h1, p=drop_p)
    h3 = lasagne.layers.SliceLayer(h2, -1, axis=1)
    h4 = lasagne.layers.DenseLayer(h3, num_units=rnn_size)
    h5 = lasagne.layers.DropoutLayer(h4, p=drop_p)
    h6 = lasagne.layers.DenseLayer(h5, num_units=out_size, nonlinearity=lasagne.nonlinearities.softmax)

    return h6


def get_train_and_val_fn(input_var, mask_var, target_var, network, lr):
    # Get final output of network
    prediction = lasagne.layers.get_output(network)
    # Calculate the loss with categorical cross entropy
    loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=target_var)
    loss = loss.mean()

    # Acquire all the parameters recursively in the network
    params = lasagne.layers.get_all_params(network, trainable=True)

    # Use default adam learning
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    # Get a deterministic output for test-time, in case we use dropout
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Get the loss according to the deterministic test-time output
    test_loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=target_var)
    test_loss = test_loss.mean()

    # Group all the inputs together
    fn_inputs = [input_var, mask_var, target_var]
    # Compile the training function
    train_fn = theano.function(fn_inputs, loss, updates=updates)
    # Compile the test function
    val_fn = theano.function(fn_inputs, test_loss)
    # compile the prediction function
    pred_fn = theano.function([input_var, mask_var], test_prediction)
    return train_fn, val_fn, pred_fn


def get_train_and_val_fn_ctc(input_var, input_lens, mask_var, output, output_lens, network, lr):
    import ctc

    # Get final output of network
    prediction = lasagne.layers.get_output(network)
    # Calculate the loss with CTC
    loss = T.mean(ctc.cpu_ctc_th(prediction, input_lens, output, output_lens))
    # Acquire all the parameters recursively in the network
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Use default adam learning
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    # # Remove NaNs from Zero or full 1 probability predictions

    # updates = replace_nans_with_zero(updates)

    # # Get a deterministic output for test-time, in case we use dropout
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # Get the loss according to the deterministic test-time output
    test_loss = T.mean(ctc.cpu_ctc_th(test_prediction, input_lens, output, output_lens))
    # Group all the inputs together
    fn_inputs = [input_var, input_lens, mask_var, output, output_lens]
    # Compile the training function
    train_fn = theano.function(fn_inputs, loss, updates=updates)
    # Compile the test function
    val_fn = theano.function(fn_inputs, test_loss)
    # compile the prediction function
    pred_fn = theano.function([input_var, mask_var], test_prediction)
    return train_fn, val_fn, pred_fn


def non_flattening_dense(l_in, batch_size, seq_len, *args, **kwargs):
    # Flatten down the dimensions for everything but the features
    l_flat = lasagne.layers.ReshapeLayer(l_in, (-1, [2]))
    # Make a dense layer connected to it
    l_dense = lasagne.layers.DenseLayer(l_flat, *args, **kwargs)
    # Reshape it back out
    l_nonflat = lasagne.layers.ReshapeLayer(l_dense, (batch_size, seq_len, l_dense.output_shape[1]))
    return l_nonflat


def get_ctc_net(input_var, mask_var, inp_dim, rnn_size, out_size, GRAD_CLIP, drop_p):
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)
    # Mask as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)
    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1 = get_dbl_gru(l_in, rnn_size, l_mask=l_mask, GRAD_CLIP=GRAD_CLIP, drop_p=drop_p)
    h2 = non_flattening_dense(h1, batch_size=batch_size, seq_len=seq_len, num_units=rnn_size,
                              nonlinearity=lasagne.nonlinearities.linear)
    h3 = non_flattening_dense(h2, batch_size=batch_size, seq_len=seq_len, num_units=out_size,
                              nonlinearity=lasagne.nonlinearities.linear)
    l_out = lasagne.layers.DimshuffleLayer(h3, (1, 0, 2))

    return l_out


def get_dbl_lstm(input_layer, rnn_size, l_mask, GRAD_CLIP, drop_p):
    # Forward layer
    hf = lasagne.layers.LSTMLayer(input_layer, rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP,
                                  hid_init=lasagne.init.GlorotUniform())

    # Backward layer
    hb = lasagne.layers.LSTMLayer(input_layer, rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP, backwards=True,
                                  hid_init=lasagne.init.GlorotUniform())

    # Concatenation
    h = lasagne.layers.ConcatLayer([hf, hb], axis=2)

    if drop_p != 'none':
        drop = lasagne.layers.DropoutLayer(h, p=drop_p)
        print('Dropout value {} used'.format(drop_p))
        return drop
    elif drop_p == 'none':
        print('No dropout used')
        return h
    else:
        print('Dropout definition error')
        return 0


def get_dbg_network(input_var, mask_var, inp_dim, rnn_size, out_size, GRAD_CLIP, drop_p):
    print('Dropout probability is {}'.format(drop_p))
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)
    # Mask as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)
    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1 = lasagne.layers.LSTMLayer(l_in, rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP,
                                  hid_init=lasagne.init.GlorotUniform())
    h2 = non_flattening_dense(h1, batch_size=batch_size, seq_len=seq_len, num_units=out_size,
                              nonlinearity=lasagne.nonlinearities.linear)
    l_out = lasagne.layers.DimshuffleLayer(h2, (1, 0, 2))

    return l_out


def replace_nans_with_zero(updates):
    # Replace all nans with zeros
    for k, v in updates.items():
        k = T.switch(T.eq(v, np.nan), float(0.), v)
    print('Warning: replaced nans')
    return updates


def get_dbl_gru(input_layer, rnn_size, l_mask, GRAD_CLIP, drop_p):
    # Forward layer
    hf = lasagne.layers.GRULayer(input_layer, rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP,
                                 hid_init=lasagne.init.GlorotUniform())

    # Backward layer
    hb = lasagne.layers.GRULayer(input_layer, rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP, backwards=True,
                                 hid_init=lasagne.init.GlorotUniform())

    # Concatenation
    h = lasagne.layers.ConcatLayer([hf, hb], axis=2)

    if drop_p != 'none':
        drop = lasagne.layers.DropoutLayer(h, p=drop_p)
        print('Dropout value {} used'.format(drop_p))
        return drop
    elif drop_p == 'none':
        print('No dropout used')
        return h
    else:
        print('Dropout definition error')
        return 0
