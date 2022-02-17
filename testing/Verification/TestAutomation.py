import numpy as np
import cv2 as cv
import pandas as pd


def GetPrecision(evaluation, output):
    blackPixels = (output < 75).all(axis=2)
    # Make sure its colored clearly
    # evaluatedPixels = (np.max(evaluation, axis=2) - np.min(evaluation, axis=2)) > 100
    evaluatedPixels = np.logical_and((evaluation > 75).any(axis=2), (evaluation < 240).any(axis=2))
    # Include for testing only the black pixels of the output image and those that are properly colored on evaluation
    includedForTesting = np.logical_and(blackPixels, evaluatedPixels)

    # Get the dominant color of each pixel of the evaluated image
    true = np.sum(np.logical_and(includedForTesting, evaluation[:, :, 0].astype(np.int16) - evaluation[:, :, 2].astype(np.int16) > 50))
    false = np.sum(np.logical_and(includedForTesting, evaluation[:, :, 2].astype(np.int16) - evaluation[:, :, 0].astype(np.int16) > 50))
    return true, false

    # # Count the number of pixels where it evalutes to blue and labeled as redundant
    # truePositive = np.sum(np.logical_and(includedForTesting, np.logical_and(evaluation==2, output>1)))
    #
    # # Count the number of pixels where it evalutes to red and labeled as redundant
    # falsePositive = np.sum(np.logical_and(includedForTesting, np.logical_and(evaluation==0, output>1)))
    #
    # # Count the number of pixels where it evalutes to blue and labeled as not redundant
    # trueNegative = np.sum(np.logical_and(includedForTesting, np.logical_and(evaluation==2, output==1)))
    #
    # # Count the number of pixels where it evalutes to red and labeled as not redundant
    # falseNegative = np.sum(np.logical_and(includedForTesting, np.logical_and(evaluation==0, output==1)))

def GetRecall(positive, evaluation, output):
    blackPixels = (output < 75).all(axis=2)
    evaluatedPixels = np.logical_and((evaluation > 75).any(axis=2), (evaluation < 240).any(axis=2))
    includedForTesting = np.logical_and(blackPixels, evaluatedPixels)

    positivePixels = np.logical_and((positive > 75).any(axis=2), (positive < 240).any(axis=2))
    positiveBlue = positive[:, :, 0].astype(np.int16) - positive[:, :, 2].astype(np.int16) > 50
    correctRedundancy = np.logical_and(positivePixels, positiveBlue)

    intersection = np.logical_and(includedForTesting, correctRedundancy)
    # evaluated - intersection = evaluated and not intersection
    not_found = np.logical_and(includedForTesting, np.logical_not(intersection))
    return np.sum(includedForTesting), np.sum(not_found)

dir_num = 1
data = []
while True:
    test = cv.imread(f'./{dir_num}/0o.jpg')
    if test is None:
        break
    img_num = 0
    while True:
        output = cv.imread(f'./{dir_num}/{img_num}o.jpg')
        positive = cv.imread(f'./{dir_num}/{img_num}p.jpg')
        negative = cv.imread(f'./{dir_num}/{img_num}n.jpg')

        if output is None:
            break

        max_height = max(output.shape[0], positive.shape[0], negative.shape[0])
        max_width = max(output.shape[1], positive.shape[1], negative.shape[1])
        if output.shape[0] != max_height or output.shape[1] != max_width:
            output = np.pad(output, [(0, max_height - output.shape[0]), (0, max_width - output.shape[1]), (0, 0)],
           mode='constant', constant_values=255)
        if positive.shape[0] != max_height or positive.shape[1] != max_width:
            positive = np.pad(positive, [(0, max_height - positive.shape[0]), (0, max_width - positive.shape[1]), (0, 0)],
           mode='constant', constant_values=255)
        if negative.shape[0] != max_height or negative.shape[1] != max_width:
            negative = np.pad(negative, [(0, max_height - negative.shape[0]), (0, max_width - negative.shape[1]), (0, 0)],
           mode='constant', constant_values=255)

        tp, fp = GetPrecision(positive, output)
        trues, fn = GetRecall(positive, negative, output)
        num_black = np.sum((output < 75).all(axis=2))
        precision, recall = tp / (tp + fp) if trues > 0 else np.nan, tp / (tp + fn) if trues > 0 else np.nan
        print("Image: {0}\nTP: {1}\nFP: {2}\nFN: {3}\ntrues: {4}\nink pixels: {5}"
                            .format(img_num, tp, fp, fn, trues, num_black, precision, recall))
        print()
        assert tp <= trues
        data.append({
            'dir_num': dir_num,
            'img_num': img_num,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'trues': trues,
            'ink_pixels': num_black,
            'precision': precision,
            'recall': recall,
        })

        # print("Image: {0}, TP + FN: {1}, FN: {2}".format(img_num, *GetRecall(positive, negative, output)))
        img_num += 1
    dir_num += 1
pd.DataFrame(data).to_excel('Data.xlsx')