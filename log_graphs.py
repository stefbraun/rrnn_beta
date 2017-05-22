import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib
import ast

matplotlib.rcParams.update({'font.size': 18})

# csv_path = 'models/dev_lr1e-4/dev_lr1e-4.csv'
csv_path = '/media/stefbraun/ext4/Dropbox/Backup/1207/Conv-Speech/wsj/models/test/log.csv'
csv_path = '/media/stefbraun/ext4/audio_group/stefan/lv_snr/models/train_si84.csv'
csv_path = '/media/stefbraun/ext4/audio_group/stefan/lv_snr/models/train_si84/train_si84.csv'
csv_path = '/media/stefbraun/ext4/audio_group/stefan/tidigits/log.csv'

plot_all = 0
category = 'seed'
keys_range = [11,12,23,30]
x_mode = 'epochs'
y_mode = 'dev_acc'

cat_dict = dict()
cat = 0
with open(csv_path, 'rb') as f:
    f = csv.reader(f)
    cat_rows = []
    for idx, row in enumerate(f):
        print(idx)
        if (idx != 0) and row[0] == 'epoch':
            cat_dict['{}'.format(cat)] = np.asarray(cat_rows)
            cat = cat + 1
            cat_rows = []
        cat_rows.append(row)
    cat_dict['{}'.format(cat)] = np.asarray(cat_rows)

if plot_all == 1:
    plt.figure()
    for key in keys_range:
        key = str(key)
        cat_idx = np.where(cat_dict[key][0] == category)[0][0]
        cat_name = cat_dict[key][1, cat_idx].astype(str)

        y_column = np.where(cat_dict[key][0] == y_mode)[0][0]
        y_values = cat_dict[key][1:, y_column].astype(float)

        # epochs
        ax = plt.subplot(311)
        x_values = cat_dict[key][1:, 0].astype(int)
        plt.plot(x_values, y_values, label=cat_name)
        ax.grid(True)
        ax.set_title('{} over epochs'.format(y_mode))
        plt.legend()

        # wct
        ax = plt.subplot(312)
        x_values = np.cumsum(cat_dict[key][1:, 8].astype(float))
        ax.plot(x_values, y_values, label=cat_name)
        ax.grid(True)
        ax.set_title('{} over wall clock time'.format(y_mode))

        # best_wct
        ax = plt.subplot(313)
        x_min = np.min(cat_dict[key][1:, 8].astype(float))
        x_values = np.arange(1, cat_dict[key][-1, 0].astype(int) + 1) * x_min
        ax.plot(x_values, y_values, label=cat_name)
        ax.grid(True)
        ax.set_title('{} over best clock time'.format(y_mode))

    plt.show()
else:
    fig, ax = plt.subplots()
    for key in keys_range:
        key = str(key)
        cat_idx = np.where(cat_dict[key][0] == category)[0][0]
        cat_name = cat_dict[key][1, cat_idx].astype(str)

        if x_mode == 'epochs':
            x_values = cat_dict[key][1:, 0].astype(int)
            unit = ' #'
        if x_mode == 'wct':
            x_values = np.cumsum(cat_dict[key][1:, 8].astype(float))
            unit = ' [sec]'
        if x_mode == 'frames':
            x_values = np.cumsum(cat_dict[key][1:, 11].astype(float))
            unit = ' [#]'
        if x_mode == 'best_wct':
            x_min = np.min(cat_dict[key][1:, 8].astype(float))
            x_values = np.arange(1, cat_dict[key][-1, 0].astype(int) + 1) * x_min
            unit = ' [sec]'
        y_column = np.where(cat_dict[key][0] == y_mode)[0][0]

        # y_values=[]
        # for i in range(len(x_values)):
        #     # y_values.append(np.max(ast.literal_eval(cat_dict[key][1:, 26][i])['ep_snr']))
        #     y_values.append(ast.literal_eval(cat_dict[key][1:, 25][i])[1])

        y_values = cat_dict[key][1:, y_column].astype(float)

        ax.plot(x_values, y_values, label=cat_name, lw=2.0)
        ax.set_ylabel(y_mode)
        ax.set_xlabel(x_mode + unit)
        ax.set_xlim((0, 150))
        # ax.set_ylim((0,1))

    plt.grid()
    plt.legend()
    plt.show()
