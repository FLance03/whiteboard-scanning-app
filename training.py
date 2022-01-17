import re
import os
import pathlib

import numpy as np
import cv2 as cv

import Steps.testing as crop_edges
import Steps.FeatureExtraction6 as get_features

def crop_images(file_names, crops):
    file_names = sorted(file_names)
    removed_edges_imgs = []
    for i in range(len(file_names)):
        img = cv.imread(file_names[i])
        before = (0, 0)
        after = (0, 0)
        assert img is not None
        removed_edges_img = []
        for x1, y1, x2, y2 in crops[i]:
            l, r = min(x1, x2), max(x1, x2)
            t, b = min(y1, y2), max(y1, y2)
            after = (t, b)
            if before != (0, 0):
                assert (before[1] - after[0]) / (after[1] - before[0]) < 0.2 or after[1] < before[0], (i, x1, y1, x2, y2)
            before = after
            cropped_img = img[t:b+1, l:r+1]
            cv.waitKey()
            removed_edges_img.append(crop_edges.main(cropped_img))
        removed_edges_imgs.append(removed_edges_img)
    return removed_edges_imgs

def get_crops_from_file(path):
    print(__file__, path)
    num_boxes = 0
    ret_crop = []
    with open(path, 'r') as f:
        while True:
            crop = f.readline()
            if crop == '':
                break
            q = re.match(r'\b(?P<x1>\d+), (?P<y1>\d+), (?P<x2>\d+), (?P<y2>\d+)\b', crop)
            if q is not None:
                # print([q['x1'], q['y1'], q['x2'], q['y2']])
                ret_crop.append([int(q['x1']), int(q['y1']),
                                 int(q['x2']), int(q['y2'])])
                num_boxes += 1
    return ret_crop, num_boxes
category_is_true = False
imgs = []
# crops = [
#     [],
#     [[131, 178, 200, 232],
#     [210, 180, 271, 232],
#     [287, 271, 404, 304]],
#     [],
# ]
crops = []
num_boxes = 0
relations = [
    # FALSE
    [0, 6], [1, 6], [2, 6],
    [0, 7], [1, 7], [2, 7],
    [0, 8], [1, 8], [2, 8],

    [3, 9], [4, 9], [5, 9],
    [3, 10], [4, 10], [5, 10],
    [3, 11], [4, 11], [5, 11],

    # TRUE
    # [0, 1], [0, 2], [1, 2],
    # [3, 4], [3, 5], [4, 5],
    # [6, 7], [6, 8], [7, 8],
    # [9, 10], [9, 11], [10, 11],
]

file_dir = './testing/training/imgs'.split('/')
parent_dir = pathlib.Path(__file__).parent.resolve()
file_dir = os.path.join(parent_dir, *file_dir[1:])
print(file_dir)
# for (_, _, filenames) in os.walk(file_dir):
#     for filename in sorted(filenames):
#         p = re.match(r'(?P<name>.*)?\.(?P<extension>\w*)', filename)
#         if p is not None and p['extension'].lower() in ['jpg', 'png']:
#             try:
#                 crop_file, num_box = get_crops_from_file(
#                     os.path.join(os.path.dirname(__file__), *'/testing/training/imgs/{}.txt'.format(p['name']).split('/')))
#             except FileNotFoundError:
#                 pass
#             else:
#                 crops.append(crop_file)
#                 num_boxes += num_box
#                 imgs.append(os.path.join(file_dir, '{0}.{1}'.format(p['name'], p['extension'])))


removed_edges_imgs = crop_images(file_names=imgs, crops=crops)

file_dir = './testing/training/crops'.split('/')
parent_dir = pathlib.Path(__file__).parent.resolve()
file_dir = os.path.join(parent_dir, *file_dir[1:])
# for img_num, img in enumerate(removed_edges_imgs):
#     for crop_num, cropped in enumerate(img):
#         cv.imwrite(os.path.join(file_dir, r'{0}-{1}.png'.format(img_num, crop_num)), cropped)
#         # cv.imshow('1', cropped)
#         # cv.waitKey()
#         # cv.destroyAllWindows()


img_crop_relations = {}
for (_, _, filenames) in os.walk(file_dir):
    for filename in filenames:
        p = re.match(r'(?P<name>.*)?\.(?P<extension>\w*)', filename)
        if p is not None and p['extension'].lower() == 'png':
            q = re.match(r'(?P<img_num>\d+)-(?P<crop_num>\d+)', p['name'])
            assert q is not None
            try:
                img_crop_relations[q['img_num']].append(q['crop_num'])
            except KeyError:
                img_crop_relations[q['img_num']] = []
                img_crop_relations[q['img_num']].append(q['crop_num'])
            # img = cv.imread(os.path.join(file_dir, filename), 0)
            # assert img is not None
            # print(get_features.main(img, labelNum=0)[0].shape)

features = []
# category = []
print(img_crop_relations)
for img_num_1, img_num_2 in relations:
    assert str(img_num_1) in img_crop_relations and str(img_num_2) in img_crop_relations, (img_num_1, img_num_2)
    for crop_num in img_crop_relations[str(img_num_1)]:
        assert crop_num in img_crop_relations[str(img_num_2)]
        img_1 = cv.imread(os.path.join(file_dir, r'{0}-{1}.png'.format(img_num_1, crop_num)), 0)
        img_2 = cv.imread(os.path.join(file_dir, r'{0}-{1}.png'.format(img_num_2, crop_num)), 0)

        min_height = min(img_1.shape[0], img_2.shape[0])
        min_width = min(img_1.shape[1], img_2.shape[1])
        img_1 = img_1[:min_height, :min_width]
        img_2 = img_2[:min_height, :min_width]
        assert img_1.shape == img_2.shape == (min_height, min_width)

        img_1 = get_features.main(img_1, labelNum=0)[0]
        img_2 = get_features.main(img_2, labelNum=0)[0]

        assert img_1.shape[2] == img_2.shape[2]
        min_height = min(img_1.shape[0], img_2.shape[0])
        min_width = min(img_1.shape[1], img_2.shape[1])
        img_1 = img_1[:min_height, :min_width, :]
        img_2 = img_2[:min_height, :min_width, :]

        img_1_non_blanks = np.logical_or(img_1[:, :, 8] != 0, img_1[:, :, 17] != 0)
        img_2_non_blanks = np.logical_or(img_2[:, :, 8] != 0, img_2[:, :, 17] != 0)
        assert img_1_non_blanks.shape == img_2_non_blanks.shape, (img_1.shape, img_2.shape)
        img_non_blanks = np.logical_or(img_1_non_blanks, img_2_non_blanks)
        img_1, img_2 = img_1[img_non_blanks], img_2[img_non_blanks]
        assert img_1.ndim == img_2.ndim == 2
        assert img_1.shape[1] == img_2.shape[1] == 18
        assert img_1.shape == img_2.shape

        diff = np.abs(img_1 - img_2)
        diff = np.concatenate((diff[:, :4], diff[:, 5:], img_1[:, 4:5], img_2[:, 4:5]), axis=1)
        features.extend(diff)

        # features_1 = np.concatenate((img_1, img_2), axis=1)
        # features_2 = np.concatenate((img_2, img_1), axis=1)
        # assert features_1.ndim == features_2.ndim == 2
        # assert features_1.shape[1] == features_2.shape[1] == 36
        # features.extend(features_1)
        # if img_num_1 != img_num_2:
        #     features.extend(features_2)
        # category.extend([category_is_true, category_is_true])

features = np.array(features)
assert features.ndim == 2 and features.shape[1] == 19
category = 1 if category_is_true else 0
features = np.concatenate((features, np.broadcast_to([category], (features.shape[0], 1))), axis=1)
assert features.ndim == 2 and features.shape[1] == 20
try:
    training_data = np.load('training_data.npz')
    assert training_data['data'].ndim == 2 and training_data['data'].shape[1] == 20
    training_data = np.concatenate((training_data['data'], features), axis=0)
    np.savez('training_data.npz', data=training_data)
except FileNotFoundError:
    np.savez('training_data.npz', data=features)

print(img_crop_relations)

