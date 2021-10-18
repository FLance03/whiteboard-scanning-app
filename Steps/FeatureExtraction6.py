from math import ceil

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from Steps import (SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5)
from testing import testing

# The number of rows in one chunks is 2*ROW_OVERLAP_SIZE and overlaps half of the rows (ROW_OVERLAP_SIZE)
ROW_OVERLAP_SIZE = 2
COL_OVERLAP_SIZE = 5
COL_WINDOW_SIZE = 10

def SimpleLinearReg(x, y):
    x, y = np.array(x), np.array(y)
    num = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    den = np.sum((x - np.mean(x)) ** 2)
    m = num / den
    b = np.mean(y) - m * np.mean(x)
    return m, b
def ExtractFeatures(connComp, labelNum, minWidth=COL_WINDOW_SIZE):
    rowSize = len(connComp)
    connComp = connComp.copy()
    connComp[np.logical_and(connComp!=0, connComp!=labelNum)] = 0
    if minWidth < COL_WINDOW_SIZE:
        minWidth = COL_WINDOW_SIZE
    if len(connComp[0]) < minWidth:
        connComp = np.concatenate((connComp,
            np.zeros((rowSize, minWidth - len(connComp[0])))
        ), axis=1)
    numExcessCol = len(connComp[0]) - COL_WINDOW_SIZE
    nonOverlapSize = COL_WINDOW_SIZE - COL_OVERLAP_SIZE
    if numExcessCol % nonOverlapSize != 0:
        remaining = numExcessCol % nonOverlapSize
        connComp = np.concatenate((connComp, np.zeros((rowSize, nonOverlapSize - remaining))), axis=1).astype(np.uint64)
    assert (len(connComp[0]) - COL_WINDOW_SIZE) % (COL_WINDOW_SIZE - COL_OVERLAP_SIZE) == 0
    stop = len(connComp[0]) - COL_WINDOW_SIZE + 1
    step = COL_WINDOW_SIZE - COL_OVERLAP_SIZE
    features = []
    for indStart in range(0, stop, step):
        features.append([])
        window = connComp[:, indStart:indStart+COL_WINDOW_SIZE]
        highestBlack = []
        lowestBlack = []
        meanBlack = []
        blackDensity = []
        numColBlack = []
        # Simple linear reg for the highest and lowest black pixels and also for their mean
        #    also records the mean black density among all the columns and the mean number of black pixels
        #    last index (each row) records whether there were black pixels (0 for no 1 for partial 2 for complete)
        #    partial only happens when there is only one column with a black pixel in which, do not do SimpleLinearReg
        for colInd in range(COL_WINDOW_SIZE):
            for rowInd in range(rowSize):
                if window[rowInd][colInd] == labelNum:
                    # Iterating from the top to bottom with this for loop starts with 0 at the highest
                    #   and increasing instead of rowSize - 1 and decreasing
                    highestBlack.append([colInd, rowSize - rowInd - 1])
                    break
            else:
                # No black pixel found in the current column, therefore, checking starting bottom is useless
                continue
            for rowInd in range(rowSize-1, -1, -1):
                # Iterate from last row going up but still, at the given index, change it
                #   to something that starts at rowSize - 1 and decreasing
                if window[rowInd][colInd] == labelNum:
                    lowestBlack.append([colInd, rowSize - rowInd - 1])
                    break
            # Get the black pixel density. Since the continue statement above was not reached, we can be assured that
            #   a new element was appended to highestBlack (which in turn appended to lowestBlack) which induced the
            #   break statement
            blackCount, totalCount = 0, 0
            for rowInd in range(rowSize - highestBlack[-1][1] - 1, rowSize - lowestBlack[-1][1]):
                # Since each index 1 column of both highestBlack and lowestBlack counts from bottom to top, transform
                #   back to counting from top to bottom. Furthermore, since it holds the index (not len), stop adds by 1
                if window[rowInd][colInd] == labelNum:
                    blackCount += 1
                totalCount += 1
            blackDensity.append(blackCount/totalCount)
            numColBlack.append(blackCount)
        if len(highestBlack) == 0:
            for i in range(6):
                # No black pixels in all columns
                # We can check the last element which holds the info of the average number of
                #   black pixels to exclude these
                features[-1].append(0)
        else:
            if len(highestBlack) > 1:
                features[-1].append(np.arctan(SimpleLinearReg(*zip(*highestBlack))[0]))
                features[-1].append(np.arctan(SimpleLinearReg(*zip(*lowestBlack))[0]))
                # The mean has the same x coords as that of highestBlack and lowestBlack for a given index
                meanBlack = (np.array(highestBlack)[:, 1] + np.array(lowestBlack)[:, 1]) / 2
                # Get the angle for the line from the simple lin reg of mean black points using the x coords of lowestBlack
                features[-1].append(np.arctan(SimpleLinearReg([x for x,y in lowestBlack], meanBlack)[0]))
            else:
                features[-1].append(0), features[-1].append(0), features[-1].append(0)
            features[-1].append(sum(blackDensity) / COL_WINDOW_SIZE)
            features[-1].append(sum(numColBlack) / COL_WINDOW_SIZE)
            features[-1].append(2) if len(highestBlack) > 1 else features[-1].append(1)
    assert indStart + 1 == stop, "Window did not exactly divide"
    features = np.array(features, dtype=np.float32)
    changeFeatures = []
    count = 0
    for ind, feature in enumerate(features):
        count = 0 if feature[-1] == 0 else count + 1
        if count == 2:
            # Since the last index is 0 if there is a black pixel else 1, only find the difference except the last index
            if features[ind, -1] == 1 or features[ind - 1, -1] == 1:
                changeFeatures.append(np.r_[features[ind, :-1] - features[ind - 1, :-1], [1]])
            else:
                changeFeatures.append(np.r_[features[ind, :-1] - features[ind - 1, :-1], [2]])
            count = 1
        else:
            changeFeatures.append([0 for _ in range(6)])
    changeFeatures = np.array(changeFeatures, dtype=np.float32)
    return features, changeFeatures


def main(labels, labelNum):
    # For all labels that match the labelNum in labels, turn it to label 1 else 0 it
    labels = np.where(labels==labelNum, 1, 0)
    # Step5.GetLabelsInfo() should return only one row of the 2d array where the first 4
    #    elements are left, right, top, and bottom of the rectangle
    assert len(Step5.GetLabelsInfo(labels)) == 1, "Two different labels in labels"
    left, right, top, bottom, *_ = labelInfo = Step5.GetLabelsInfo(labels)[0]
    start = top
    # The number of rows in one chunks is 2*ROW_OVERLAP_SIZE and overlaps half of the rows (ROW_OVERLAP_SIZE)
    stop = bottom - 2*(ROW_OVERLAP_SIZE - 1)
    features = []
    for i in range(start, stop, ROW_OVERLAP_SIZE):
        # Get the features per chunks of 2*ROW_OVERLAP_SIZE rows with ROW_OVERLAP_SIZE rows overlapping
        featureData = ExtractFeatures(labels[i:i + 2*ROW_OVERLAP_SIZE, left:right+1], 1)
        features.append(np.concatenate((featureData[0], featureData[1]), axis=1))
    if stop <= start:
        # The loop did not run even once
        featureData = ExtractFeatures(labels[top:bottom+1, left:right+1], labelNum)
        features.append(np.concatenate((featureData[0], featureData[1]), axis=1))
    elif i + 1 < stop:
        # The loop ran at least once but was not able to cover all (some of the last elements
        #   were not processed)
        # lastInd = top + (stop - start - 1) // ROW_OVERLAP_SIZE * ROW_OVERLAP_SIZE
        featureData = ExtractFeatures(labels[i+ROW_OVERLAP_SIZE:bottom+1, left:right+1], 1)
        features.append(np.concatenate((featureData[0], featureData[1]), axis=1))
    features = np.array(features, dtype=np.float32)
    return features, labelInfo

# img = cv.imread('./testing/pics and texts/iotbinarized.jpg', 0)
# labels, labelsInfo, textNonText, textLabels, wordLabels, phraseLabels, nonTextLabels = Step5.main(img)
# np.savez('Step5Outputs.npz', labels=labels, labelsInfo=labelsInfo, textNonText=textNonText, textLabels=textLabels,
#          wordLabels=wordLabels, phraseLabels=phraseLabels, nonTextLabels=nonTextLabels)
# file = np.load('Step5Outputs.npz')
# labels, labelsInfo, textNonText, textLabels, wordLabels, phraseLabels, nonTextLabels = \
#     file['labels'], file['labelsInfo'], file['textNonText'], file['textLabels'], file['wordLabels'], file[
#         'phraseLabels'], file['nonTextLabels']
# labelInfo = labelsInfo[138]
# testing.FullPrint(labels[top:bottom+1, left:right+1])
# testing.FullPrint(labelsInfo)

# cv.imshow('Labels', testing.imshow_components(labels))
# cv.imshow('Texts', testing.imshow_components(textLabels))
# cv.imshow('Words', testing.imshow_components(wordLabels))
# cv.imshow('Phrases', testing.imshow_components(phraseLabels))
# cv.imshow('Non Texts', testing.imshow_components(nonTextLabels))
#
# cv.waitKey()
# cv.destroyAllWindows()
