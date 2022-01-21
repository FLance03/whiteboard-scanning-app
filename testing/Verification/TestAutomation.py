import numpy as np
import cv2 as cv


def GetPrecision(evaluation, output):
    blackPixels = (output < 75).all(axis=2)
    # Make sure its colored clearly
    # evaluatedPixels = (np.max(evaluation, axis=2) - np.min(evaluation, axis=2)) > 100
    evaluatedPixels = np.logical_and((evaluation > 75).any(axis=2), (evaluation < 200).any(axis=2))
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
    return np.sum(includedForTesting), np.sum(blackPixels), np.sum(not_found)


img_num = 0
while True:
    output = cv.imread(str(img_num) + 'o.jpg')
    positive = cv.imread(str(img_num) + 'p.jpg')
    negative = cv.imread(str(img_num) + 'n.jpg')
    if output is None:
        break
    tp, fp = GetPrecision(positive, output)
    trues, num_black, fn = GetRecall(positive, negative, output)
    precision, recall = tp / (tp + fp) if trues > 0 else np.nan, tp / (tp + fn) if trues > 0 else np.nan
    print("Image: {0}\npercent TP: {1}\npercent FP: {2}\npercent FN: {3}\ntrues: {4}\npercent true: {5}\nprecision: {6}\nrecall: {7}"
                        .format(img_num, tp/num_black, fp/num_black, fn/num_black, trues, trues/num_black, precision, recall))
    print()
    # print("Image: {0}, TP + FN: {1}, FN: {2}".format(img_num, *GetRecall(positive, negative, output)))
    img_num += 1