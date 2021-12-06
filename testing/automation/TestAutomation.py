import numpy as np
import cv2 as cv


def GetErrors(evaluation, output):
    blackPixels = output > 0
    # Make sure its colored clearly
    evaluatedPixels = (np.max(evaluation, axis=2) - np.min(evaluation, axis=2)) > 100
    # Include for testing only the black pixels of the output image and those that are properly colored on evaluation
    includedForTesting = np.logical_and(blackPixels, evaluatedPixels)

    # Get the dominant color of each pixel of the evaluated image
    evaluation = np.argmax(evaluation, axis=2)

    # Count the number of pixels where it evalutes to blue and labeled as redundant
    truePositive = np.sum(np.logical_and(includedForTesting, np.logical_and(evaluation==2, output>1)))

    # Count the number of pixels where it evalutes to red and labeled as redundant
    falsePositive = np.sum(np.logical_and(includedForTesting, np.logical_and(evaluation==0, output>1)))

    # Count the number of pixels where it evalutes to blue and labeled as not redundant
    trueNegative = np.sum(np.logical_and(includedForTesting, np.logical_and(evaluation==2, output==1)))

    # Count the number of pixels where it evalutes to red and labeled as not redundant
    falseNegative = np.sum(np.logical_and(includedForTesting, np.logical_and(evaluation==0, output==1)))

evaluation = cv.imread('e.jpg')
output = cv.imread('o.jpg')

print("TP: {0}, FP: {1}, TN: {2}, FN: {3}".format(*GetErrors(evaluation, output)))
