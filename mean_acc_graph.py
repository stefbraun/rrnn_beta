import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib
import ast

matplotlib.rcParams.update({'font.size': 18})

# csv_path = 'models/dev_lr1e-4/dev_lr1e-4.csv'
csv_path = '/media/stefbraun/ext4/Dropbox/repos/tidigits/log_super.csv'


plot_all = 0
category = 'seed'
keys_range = [0,1,2,3,4,5]
y_mode = 'dacc'

cat_dict = dict()
cat = 0
with open(csv_path, 'rb') as f:
    f = csv.reader(f)
    cat_rows = []
    for idx, row in enumerate(f):
        print(idx)
        if (idx != 0) and row[0] == 'seed':
            cat_dict['{}'.format(cat)] = np.asarray(cat_rows)
            cat = cat + 1
            cat_rows = []
        cat_rows.append(row)
    cat_dict['{}'.format(cat)] = np.asarray(cat_rows)

bar_x = []
bar_y = []
bar_std = []
width = 0.25
for key in keys_range:
    key = str(key)
    cat_idx = np.where(cat_dict[key][0] == category)[0][0]
    cat_name = cat_dict[key][1, cat_idx].astype(float)

    y_column = np.where(cat_dict[key][0] == y_mode)[0][0]
    y_values = cat_dict[key][1:, y_column].astype(float)

    bar_x.append(cat_name)
    bar_y.append(np.mean(y_values))
    bar_std.append(np.std(y_values))

fig, ax = plt.subplots(1)
ax.bar(bar_x, bar_y, width=width, color='lightgray', yerr=bar_std, error_kw={'ecolor':'blue',
                          'linewidth':2})
ax.grid(True)
ax.set_title('{} over thresholds, mean and std of 10 runs'.format(y_mode))
ax.set_xlim([0, 5])
ax.set_ylabel('Accuracy [%]')
ax.set_xticks(np.asarray(bar_x)+width/2)
ax.set_xticklabels(bar_x)


print(bar_x)
print(bar_y)
plt.legend()
plt.show()
