import numpy as np
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper

from lasagne.layers.recurrent import Gate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng


class RRNNTimeGate(object):
    """
    """

    def __init__(self,
                 Period=init.Uniform((10, 100))):
        self.Period = Period

class RRNNLayer(MergeLayer):
    r"""
    """

    # GATE defaults: W_in=init.Normal(0.1), W_hid=init.Normal(0.1), W_cell=init.Normal(0.1), b=init.Constant(0.), nonlinearity=nonlinearities.sigmoid
    def __init__(self, incoming, time_input, num_units,
                 W_in=lasagne.init.GlorotUniform(),
                 W_hid=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 timegate=RRNNTimeGate(),
                 hid_init=lasagne.init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 bn=False,
                 learn_time_params=[True, True, False],
                 off_alpha=1e-3,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        # TIME STUFF
        if time_input != None:
            incomings.append(time_input)
            self.time_incoming_index = len(incomings) - 1
        else:
            print('Self-generating time input!')
            self.time_incoming_index = None

        self.mask_incoming_index = -2
        self.hid_init_incoming_index = -2
        # self.cell_init_incoming_index = -2

        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1
        # if isinstance(cell_init, Layer):
        #     incomings.append(cell_init)
        #     self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(RRNNLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        time_shape = self.input_shapes[1]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        # def add_params(gate, gate_name):
        #     """ Convenience function for adding layer parameters from a Gate
        #     instance. """
        #     return (self.add_param(gate.W_in, (num_inputs, num_units),
        #                            name="W_in_to_{}".format(gate_name)),
        #             self.add_param(gate.W_hid, (num_units, num_units),
        #                            name="W_hid_to_{}".format(gate_name)),
        #             self.add_param(gate.b, (num_units,),
        #                            name="b_{}".format(gate_name),
        #                            regularizable=False),
        #         gate.nonlinearity)

        # PHASED LSTM: Initialize params for the time gate
        self.off_alpha = off_alpha
        if timegate == None:
            timegate = RRNNTimeGate()

        def add_timegate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.Period, (num_units,),
                                   name="Period_{}".format(gate_name),
                                   trainable=learn_time_params[0]))
            # self.add_param(gate.Shift, (num_units, ),
            #                name="Shift_{}".format(gate_name),
            #                trainable=learn_time_params[1]),
            # self.add_param(gate.On_End, (num_units, ),
            #                name="On_End_{}".format(gate_name),
            #                trainable=learn_time_params[2]))

        print('Learnableness: {}'.format(learn_time_params))
        (self.period_timegate) = add_timegate_params(timegate, 'timegate')

        self.W_in = self.add_param(W_in,
                                   (num_inputs, num_units), name="W_in")
        self.W_hid = self.add_param(W_hid, (num_units, num_units), name="W_hid")
        self.b = self.add_param(b, (num_units,), name="b")

        # Add in parameters from the supplied Gate instances
        # (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
        #  self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        #
        # (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
        #  self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
        #                                                  'forgetgate')
        #
        # (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
        #  self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        #
        # (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
        #  self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        # if self.peepholes:
        #     self.W_cell_to_ingate = self.add_param(
        #         ingate.W_cell, (num_units, ), name="W_cell_to_ingate")
        #
        #     self.W_cell_to_forgetgate = self.add_param(
        #         forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")
        #
        #     self.W_cell_to_outgate = self.add_param(
        #         outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        # if isinstance(cell_init, Layer):
        #     self.cell_init = cell_init
        # else:
        #     self.cell_init = self.add_param(
        #         cell_init, (1, num_units), name="cell_init",
        #         trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        if bn:
            self.bn = lasagne.layers.BatchNormLayer(input_shape, axes=(0, 1))  # create BN layer for correct input shape
            self.params.update(self.bn.params)  # make BN params your params
        else:
            self.bn = False

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # PHASED LSTM: Define new input
        if self.time_incoming_index != None:
            # Pull off the layer if it is real
            time_mat = inputs[self.time_incoming_index]
        else:
            # Otherwise autogenerate a timestep
            if self.backwards:
                time_mat = T.arange(input.shape[1])[::-1].dimshuffle('x', 0).astype('float32')
            else:
                time_mat = T.arange(input.shape[1]).dimshuffle('x', 0).astype('float32')

        if self.bn:
            input = self.bn.get_output_for(input)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        # PHASED LSTM: Get shapes for time input and rearrange for the scan fn
        time_input = time_mat.dimshuffle(1, 0)
        time_seq_len, time_num_batch = time_input.shape
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        # W_in_stacked = T.concatenate(
        #     [self.W_in_to_ingate, self.W_in_to_forgetgate,
        #      self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # PHASED LSTM: If test time, off-phase means really shut.
        if deterministic:
            print('Using true off for testing.')
            off_slope = 0.0
        else:
            print('Using {} for off_slope.'.format(self.off_alpha))
            off_slope = self.off_alpha

        # PHASED LSTM: Pregenerate broadcast vars.
        #   Same neuron in different batches has same shift and period.  Also,
        #   precalculate the middle (on_mid) and end (on_end) of the open-phase
        #   ramp.
        # shift_broadcast = self.shift_timegate.dimshuffle(['x',0])
        period_broadcast = T.abs_(self.period_timegate.dimshuffle(['x', 0]))
        # on_mid_broadcast = T.abs_(self.on_end_timegate.dimshuffle(['x',0])) * 0.5 * period_broadcast
        # on_end_broadcast = T.abs_(self.on_end_timegate.dimshuffle(['x',0])) * period_broadcast

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, self.W_in) + self.b
        ########################################################################################## 4:00pm

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        # def slice_w(x, n):
        #     return x[:, n * self.num_units:(n + 1) * self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input

        # def step(input_n, time_input_n, cell_previous, hid_previous, *args):
        def step(input_n, time_input_n, hid_previous, *args):

            if not self.precompute_input:
                input_n = T.dot(input_n, self.W_in) + self.b  # modified

            # Calculate gates pre-activations and slice
            # gates = input_n + T.dot(hid_previous, W_hid_stacked)
            states = input_n + T.dot(hid_previous, self.W_hid)

            # Clip gradients
            if self.grad_clipping:
                states = theano.gradient.grad_clip(
                    states, -self.grad_clipping, self.grad_clipping)

            hid = self.nonlinearity(states)

            return hid

        # RRNN: The actual calculation of the time gate
        def calc_time_gate(time_input_n, refrac_end_prev):
            # Broadcast the time across all units
            t_broadcast = time_input_n.dimshuffle([0, 'x'])
            # Find what's allowed to update
            sleep_wake_mask = T.gt(t_broadcast, refrac_end_prev)
            # Update the refractory period
            new_refrac_end = T.switch(sleep_wake_mask, t_broadcast + period_broadcast, refrac_end_prev)
            return sleep_wake_mask, new_refrac_end

        def step_masked(input_n, time_input_n, mask_n, hid_previous, refrac_end_prev, *args):
            # cell, hid = step(input_n, time_input_n, cell_previous, hid_previous, *args)
            hid = step(input_n, time_input_n, hid_previous, *args)

            # Get time gate openness
            sleep_wake_mask, refrac_end_new = calc_time_gate(time_input_n, refrac_end_prev)

            # Sleep if off, otherwise stay a bit on
            # cell = sleep_wake_mask*cell + (1.-sleep_wake_mask)*cell_previous
            hid = sleep_wake_mask * hid + (1. - sleep_wake_mask) * hid_previous

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            # cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [hid, refrac_end_new]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
        else:
            mask = T.ones_like(time_input).dimshuffle(0, 1, 'x')

        sequences = [input, time_input, mask]
        step_fun = step_masked

        ones = T.ones((num_batch, 1))
        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        non_seqs = [self.W_hid, self.period_timegate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            # non_seqs += [W_in_stacked, b_stacked]
            non_seqs += [self.W_in, self.b]

        refrac_end_init = T.dot(ones*0., self.hid_init)
        if self.unroll_scan:
            print('please disable unroll_scan')
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init, refrac_end_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out, refrac_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init, refrac_end_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            refrac_out = refrac_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        # Store for debugging
        self.refrac_out = refrac_out

        return hid_out