import numpy as np
import cv2 as cv


def GetErrors(evaluation, output):
    blackPixels = (output == 0).all(axis=2)
    # Make sure its colored clearly
    # evaluatedPixels = (np.max(evaluation, axis=2) - np.min(evaluation, axis=2)) > 100
    evaluatedPixels = np.logical_and((evaluation > 75).any(axis=2), (evaluation < 250).any(axis=2))
    # Include for testing only the black pixels of the output image and those that are properly colored on evaluation
    includedForTesting = np.logical_and(blackPixels, evaluatedPixels)

    # Get the dominant color of each pixel of the evaluated image
    evaluation = np.argmax(evaluation, axis=2)

    true = np.sum(np.logical_and(includedForTesting, evaluation == 0))
    false = np.sum(np.logical_and(includedForTesting, evaluation == 2))
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

img_num = 0
while True:
    output = cv.imread(str(img_num) + 'o.jpg')
    positive = cv.imread(str(img_num) + 'p.jpg')
    negative = cv.imread(str(img_num) + 'n.jpg')
    if output is None:
        break
    print("Image: {0}, TP: {1}, FP: {2}".format(img_num, *GetErrors(positive, output)))
    print("Image: {0}, TN: {1}, FN: {2}".format(img_num, *GetErrors(negative, output)))
    img_num += 1