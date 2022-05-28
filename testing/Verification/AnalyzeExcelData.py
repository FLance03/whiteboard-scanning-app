import warnings

import numpy as np
import pandas as pd


df = pd.read_excel('cope.xlsx')

data = df.to_numpy()
columns = {
    'video_num': 0, '15min_num': 1, 'img_num': 2, 'TP': 3, 'FP': 4, 'FN': 5, 'trues': 6, 'ink_pixels': 7
}

percents = data[:, 3:7] / data[:, 7:8]
percents = np.concatenate((data[:, :3], percents), axis=1)

FP_percents = percents[:, columns['FP']]
trues_percents = percents[:, columns['trues']]

percent_bins = np.array(np.linspace(0.1, 1, 10))
trues_binned = np.searchsorted(percent_bins, trues_percents, side='left')
assert 10 not in trues_binned
y_axis = []
for i in range(10):
    try:
        y_axis.append(np.mean(FP_percents[trues_binned == i]))
    except RuntimeWarning:
        y_axis.append(np.nan)
print('1:', y_axis)

max_img_num = np.max(data[:, columns['img_num']])
precision = data[:, columns['TP']] / (data[:, columns['TP']] + data[:, columns['FP']])
recall = data[:, columns['TP']] / (data[:, columns['TP']] + data[:, columns['FN']])
print('Precision: ', precision)
print('Recall: ', recall)

warnings.filterwarnings("error")

y_axis = []
for i in range(max_img_num + 1):
    img_num_inds = data[:, columns['img_num']] == i
    try:
        y_axis.append(np.mean(precision[img_num_inds]))
    except RuntimeWarning:
        y_axis.append(np.nan)
print('2 (Precision): ', y_axis)
y_axis = []
for i in range(max_img_num + 1):
    img_num_inds = data[:, columns['img_num']] == i
    try:
        y_axis.append(np.mean(precision[img_num_inds]))
    except RuntimeWarning:
        y_axis.append(np.nan)
print('2 (Recall): ', y_axis)

img_counts = np.bincount(data[:, columns['15min_num']].astype(np.int32))
max_num_imgs = np.max(img_counts)
y_axis = [np.nan, np.nan]
for i in range(2, max_num_imgs + 1):
    fift_min_inds_with_count = img_counts == i
    num = 0
    sum = 0
    for fift_min_ind_num_with_count, fift_min_ind_with_count in enumerate(fift_min_inds_with_count):
        if fift_min_ind_with_count:
            fift_min_inds = data[:, columns['15min_num']] == fift_min_ind_num_with_count
            sum += np.sum(precision[fift_min_inds])
            num += i
    if num > 0:
        y_axis.append(sum / num)
    else:
        y_axis.append(np.nan)
print('3 (Precision): ', y_axis)
y_axis = [np.nan, np.nan]
for i in range(2, max_num_imgs + 1):
    fift_min_inds_with_count = img_counts == i
    num = 0
    sum = 0
    for fift_min_ind_num_with_count, fift_min_ind_with_count in enumerate(fift_min_inds_with_count):
        if fift_min_ind_with_count:
            fift_min_inds = data[:, columns['15min_num']] == fift_min_ind_num_with_count
            sum += np.sum(recall[fift_min_inds])
            num += i
    if num > 0:
        y_axis.append(sum / num)
    else:
        y_axis.append(np.nan)
print('3 (Recall): ', y_axis)
