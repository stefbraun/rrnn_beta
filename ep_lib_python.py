from __future__ import print_function
import numpy as np
import theano.sandbox.cuda
import h5py
import warnings
# from preprocess_lib import get_socket_from_gpu
import zmq

class EpochIterator():
    def flow(self, start_epoch, num_epochs, max_patience, schedule, ep_type, SNR, train_dataset, dev_dataset, debug):

        # Warnings
        if ep_type not in ['baseline', 'gauss', 'pem', 'gauss_pem', 'curriculum']:
            warnings.warn('Unknown ep_type {}'.format(ep_type))
            return

        if debug != 0:
            warnings.warn('Epoch iterator is in debug mode!')

        # Continuation of training: hack around start epoch, consider that baseline and gauss always use first epoch!
        if start_epoch != 1:
            warnings.warn(
                'Start epoch is {} instead of 1. Make sure this is the continuation of an existing experiment.'.format(
                    start_epoch))
            if (ep_type == 'baseline') or (ep_type == 'gauss'):
                train_epoch = 1
            else:
                train_epoch = start_epoch
        else:
            train_epoch = start_epoch

        # Initialize parameters
        epochs = start_epoch - 1
        self.mon_var = [np.Inf]
        self.patience = 0
        c_stage = 0  # curriculum stage of SNR
        curr_snr = SNR[0]
        switch_stage = 0
        best_epoch = 0

        # Start preprocessor processes and get first epoch of features
        if debug == 0:
            gpu = theano.sandbox.cuda.config.device
        else:
            gpu = 'test'
        socket = get_socket_from_gpu(gpu)
        context = zmq.Context()
        sock = context.socket(zmq.REQ)
        sock.connect("tcp://127.0.0.1:{}".format(socket))
        work = [(train_dataset, train_epoch, curr_snr, gpu, 'train', debug),
                (dev_dataset, 1, curr_snr, gpu, 'dev', debug)]
        sock.send_pyobj(work)

        # Epoch loop
        while epochs < num_epochs:

            req_switch = 0
            self.load_best_model = (-5, -5)

            # Wait for data preprocessor, then get result
            paths = sock.recv_pyobj()

            if 'train' in paths:
                train_out = paths['train']
            if 'dev' in paths:
                dev_out = paths['dev']
            work = []

            epochs += 1

            # if more than one epoch done, start patience monitoring
            if (len(self.mon_var) > 1):

                # Update patience
                # if (self.mon_var[-1] < self.mon_var[-2]):
                if (self.mon_var[-1] < np.min(self.mon_var[:-1])):
                    self.patience = 0
                else:
                    self.patience = self.patience + 1

                # Track best model
                if (self.mon_var[-1] < np.min(self.mon_var[:-1])):
                    best_epoch = epochs-1 # we already incremented the epoch counter, but this is actually the last epoch

                # Enforce patience or schedule
                if ep_type == 'curriculum':
                    if (schedule[0] == 1) and (epochs % schedule[1] == 0):
                        req_switch = 1
                    elif (schedule[0] == 0) and (self.patience > max_patience):
                        req_switch = 1

                    if req_switch == 1:  # if switch is requested by either patience or scheduling
                        if len(SNR) > 1:  # if this was not last SNR stage

                            # Update remaining SNR stages
                            SNR = SNR[1:]
                            curr_snr = SNR[0]
                            c_stage += 1

                            # re-initalize patience and monitoring variable for next stage
                            self.patience = 0
                            self.mon_var = [np.Inf]
                            switch_stage = 1

                            # reinitalize network to last stages best weights
                            self.load_best_model = (1, best_epoch)

                        else:  # as this was last SNR stage, stop training
                            print('Training is now in last curriculum stage.')
                            # break

                elif self.patience > max_patience:  # stop training because patience was reached
                    print('Training stopped because patience has been reached.')
                    break

            # Handle switch of snr stage
            if (ep_type == 'curriculum') and (switch_stage == 1):
                print('Switching snr stage, thus we have to wait for data.')
                work.append((train_dataset, epochs, curr_snr, gpu, 'train', debug))
                work.append((dev_dataset, 1, curr_snr, gpu, 'dev{}'.format(c_stage), debug))
                sock.send_pyobj(work)

                # Wait for data preprocessor, then get result
                paths = sock.recv_pyobj()
                if 'train' in paths:
                    train_out = paths['train']
                if 'dev{}'.format(c_stage) in paths:
                    dev_out = paths['dev{}'.format(c_stage)]
                work = []

                # Reset switch stage
                switch_stage = 0

            # Remix training set for next epoch if required by ep_type, only do this when this is not the last epoch!
            if epochs < num_epochs:
                if ep_type in ['pem', 'gauss_pem', 'curriculum']:
                    work.append((train_dataset, epochs + 1, curr_snr, gpu, 'train', debug))

            # Enable gaussian noise addition for select ep_types
            if ep_type in ['gauss', 'gauss_pem', 'curriculum']:
                enable_gauss = 1
            else:
                enable_gauss = 0

            # Monitor variables
            ep_monitor = {'ep_type': ep_type, 'patience': self.patience, 'ep_snr': curr_snr, 'epochs': epochs,
                          'c_stage': c_stage, 'best_epoch': best_epoch, 'req_switch': req_switch}

            sock.send_pyobj(work)

            yield epochs, train_out, dev_out, ep_monitor, enable_gauss


def checksum(h5, type):
    hf = h5py.File(h5, 'r')

    data = hf.get(type)
    if type == 'snr':
        data = [np.min(data), np.max(data)]
    csum = np.sum(data)
    return csum


def get_crc_snr(high, low, step):
    # get curriculum style snr list, low SNR to high SNR
    full_range = range(high, low, step)
    full_range_flip = np.flipud(full_range)
    crc_snr = []
    for element in full_range_flip:
        crc_snr.append(range(element, low, step))
    return crc_snr

class SimpleEpochIterator():
    def flow(self, start_epoch, num_epochs, max_patience, ep_type, train_dataset, dev_dataset, debug):

        # Warnings
        if debug != 0:
            warnings.warn('Epoch iterator is in debug mode!')

        # Continuation of training: hack around start epoch, consider that baseline and gauss always use first epoch!
        if start_epoch != 1:
            warnings.warn(
                'Start epoch is {} instead of 1. Make sure this is the continuation of an existing experiment.'.format(
                    start_epoch))

        # Initialize parameters
        epochs = start_epoch - 1
        self.mon_var = [-np.Inf]
        self.patience = 0
        best_epoch = 0

        # Epoch loop
        while epochs < num_epochs:

            train_out = train_dataset
            dev_out = dev_dataset

            epochs += 1

            # if more than one epoch done, start patience monitoring
            if (len(self.mon_var) > 1):

                # Update patience
                # if (self.mon_var[-1] < self.mon_var[-2]):
                if (self.mon_var[-1] > np.max(self.mon_var[:-1])):
                    self.patience = 0
                else:
                    self.patience = self.patience + 1

                # Track best model
                if (self.mon_var[-1] > np.max(self.mon_var[:-1])):
                    best_epoch = epochs-1 # we already incremented the epoch counter, but this is actually the last epoch

                # Enforce patience
                if self.patience > max_patience:  # stop training because patience was reached
                    print('Training stopped because patience has been reached.')
                    break

            # Enable gaussian noise addition for select ep_types
            if ep_type in ['gauss']:
                enable_gauss = 1
            else:
                enable_gauss = 0

            # Monitor variables
            ep_monitor = {'ep_type': ep_type, 'patience': self.patience, 'epochs': epochs,
                          'best_epoch': best_epoch}

            yield epochs, train_out, dev_out, ep_monitor, enable_gauss