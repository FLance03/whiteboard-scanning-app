import os
import pathlib
import re

import numpy as np
import cv2 as cv

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=cv.INTER_AREA)

# imgs = np.load('output.npz')['data']
#
# def imshow_components(labels):
#     if np.max(labels) == 0:
#         return labels.astype(np.uint8)
#     # Map component labels to hue val
#     label_hue = np.uint8(179*labels/np.max(labels))
#     blank_ch = 255*np.ones_like(label_hue)
#     labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
#
#     # cvt to BGR for display
#     labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
#
#     # set bg label to black
#     labeled_img[label_hue==0] = 0
#
#     return labeled_img
# print(np.unique(imgs))
# print(imgs.shape)
# cv.imshow('1', imshow_components(imgs[2]))
# cv.waitKey()

output = np.load('output.npz')
labels = output['labels']
colors = output['bgr']
# parent_dir = pathlib.Path(__file__).parent.resolve()
# imgs = []
# for _, _, filenames in os.walk(parent_dir):
#     print(sorted(filenames))
#     for filename in sorted(filenames):
#         p = re.match(r'(?P<name>.*)?\.(?P<extension>\w*)', filename)
#         if p is not None and p['extension'].lower() in ['jpg', 'png']:
#             imgs.append(cv.imread(filename))

# assert [img.shape for img in imgs].count(imgs[0].shape) == len(imgs), 'some imgs have different shapes'
# assert len(imgs) % 2 == 0

# current = np.array(labels[:len(labels)//2])
# past = np.array(labels[len(labels)//2:])
for current_num, current_img_group in enumerate(labels):
    if current_num == 0:
        continue
    # Get current image from its group
    current_img = current_img_group[current_num]
    # Get unique elements and is already sorted
    current_labels = np.unique(current_img)
    assert current_labels[0] == 0 
    current_labels = current_labels[1:] 
    for current_label in current_labels:
        filter_img = current_img == current_label
        # Get 3d black/white image of current
        cleaned_current_color = colors[current_num]
        # (cleaned_current_color != [0, 0, 0]).all(axis=2) outputs a 2d boolean array and is used to index a 3d array 
        cleaned_current_color[(cleaned_current_color != [0, 0, 0]).any(axis=2)] = [255, 255, 255]
        cleaned_current_color[np.logical_and((cleaned_current_color == [255, 255, 255]).all(axis=2), filter_img)] = [0, 255, 0]
        past = current_img_group
        for past_num in range(current_num):
            test_num = 0
            if past_num != test_num and current_num != test_num:
                continue
            past_img = past[past_num]
            if (past_img == current_label).any():
                filter_past = past_img == current_label
                cleaned_past_color = colors[past_num]
                cleaned_past_color[(cleaned_past_color != [0, 0, 0]).any(axis=2)] = [255, 255, 255]
                cleaned_past_color[np.logical_and((cleaned_past_color == [255, 255, 255]).all(axis=2), filter_past)] = [0, 255, 0]
            #     np.logical_or((current_img == [0, 0, 0]).all(axis=2),
            # np.logical_or((current_img == [255, 255, 255]).all(axis=2), (current_img == color).all(axis=2)))
                # (current_img == color).all(axis=2) returns a 2d array from a 3d so add new axis in filter_img
                cv.imshow("" + str(current_num), ResizeWithAspectRatio(cleaned_current_color.astype(np.uint8), height=500))
                cv.imshow("" + str(past_num), ResizeWithAspectRatio(cleaned_past_color.astype(np.uint8), height=500))
                print("Current: {}, Past: {}, Label Num: {}".format(current_num, past_num, current_label))
                cv.waitKey()
                cv.destroyAllWindows()