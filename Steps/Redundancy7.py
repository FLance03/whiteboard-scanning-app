from itertools import combinations as comb
import heapq
from time import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import svm
import joblib

from Steps import (SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5,
                   FeatureExtraction6 as Step6)
from testing import testing

ROW_OVERLAP_SIZE = 3
ROW_WINDOW_SIZE = ROW_OVERLAP_SIZE * 2
COL_OVERLAP_SIZE = 5
COL_WINDOW_SIZE = 10

MINHEIGHTPROP = 0.4
# THRESHOLD1 is the threshold where only when the sum of squares is greater than it that it will consider something redundant
THRESHOLD1 = 400
# THRESHOLD2 gives the minimum width allowed for redundancy
THRESHOLD2 = 2
THRESHOLD3 = 2/5

count = 0
countMerged = [0, 0, 0, 0]
sumTime = 0

# For testing only
lapTime = time()
sumTime = sumTime + time() - lapTime
count += 1

def TestCompatibilityOfTwoFeatures(first, second):
    global fit
    assert first.shape == second.shape, "hi"
    assert first.ndim == 3, str(first.ndim)
    assert first.shape[2] == 18

    img_1, img_2 = first.reshape(-1, 18), second.reshape(-1, 18)
    output = np.ones((img_1.shape[0]), dtype=np.uint8)
    img_1_non_blanks = np.logical_or(img_1[:, 8] != 0, img_1[:, 17] != 0)
    img_2_non_blanks = np.logical_or(img_2[:, 8] != 0, img_2[:, 17] != 0)
    img_non_blanks = np.logical_or(img_1_non_blanks, img_2_non_blanks)
    non_blank_inds = np.asarray(img_non_blanks).nonzero()[0]
    if non_blank_inds.size > 0:
        img_1, img_2 = img_1[img_non_blanks], img_2[img_non_blanks]
        diff = np.abs(img_1 - img_2)
        diff = np.concatenate((diff[:, :4], diff[:, 5:], img_1[:, 4:5], img_2[:, 4:5]), axis=1)
        # Scale to preprocess
        diff[:, (0, 1, 2, 8, 9, 10)] = (diff[:, (0, 1, 2, 8, 9, 10)] - 0) / (60 - 0)
        diff[:, (12)] = (diff[:, (12)]) / (6 - 0)
        diff[:, (4, 13)] = (diff[:, (4, 13)]) / (2.5 - 0)
        diff[:, (5, 14)] = (diff[:, (5, 14)]) / (3 - 0)
        diff[:, (6, 15)] = (diff[:, (6, 15)]) / (10 - 0)
        diff[:, (7, 16)] = (diff[:, (7, 16)]) / (2 - 0)
        diff[:, (17, 18)] = (diff[:, (17, 18)]) / (6 - 0)

        # start_pred = time()
        output[non_blank_inds] = np.where(fit.predict(diff) == 1, 2, 0)
        # print(time() - start_pred)
    # print('output: ', output[:10])
    output = output.reshape(*first.shape[:-1])

    return output


def GetWeight(redundancyMatrix):
    imgMatrix = np.where(redundancyMatrix==0, 0, 255)
    numLabels, labels = cv.connectedComponents(imgMatrix.astype(np.uint8), connectivity=8)
    labelsInfo = Step5.GetLabelsInfo(labels, numLabels)
    assert labelsInfo.ndim == 2
    labelWeights = np.zeros((numLabels), np.uint64)
    # Find total weights for each label
    uniques, counts = np.unique(labels[redundancyMatrix == 2], return_counts=True)
    labelWeights[uniques] = counts**2
    # Concatenate label number information before sorting
    labelsInfo = np.concatenate((labelsInfo, np.arange(1, len(labelsInfo)+1).reshape(-1, 1)), axis=1)
    # Sort by the left side of the box (labels)
    sortInd = np.argsort(labelsInfo[:, 0])
    leftSort = labelsInfo[sortInd]
    # Sort by the right side of the box (labels)
    sortInd = np.argsort(labelsInfo[:, 1])
    rightSort = labelsInfo[sortInd]
    rightInd = 0
    leftInd = 0
    retVal = {
        'weight': 0,
        'left': 0,
        'right': 0,
    }
    while leftInd < len(labelsInfo):
        sum = 0
        while rightInd < len(labelsInfo) and leftSort[leftInd, 0] >= rightSort[rightInd, 1]:
            rightInd += 1
        # Since we need to remember where we left off on rightInd, traverse the rest of array using another variable
        travRightInd = rightInd
        top, bottom = len(redundancyMatrix), 0
        while travRightInd < len(labelsInfo):
            # It may happen that the right end point is within the current offsets to be checked but the
            #    left end point is not so only include in sum when the below condition is true
            if rightSort[travRightInd, 0] >= leftSort[leftInd, 0]:
                sum += labelWeights[rightSort[travRightInd, 10]]
                top = min(top, leftSort[leftInd, 2], rightSort[travRightInd, 2])
                bottom = max(bottom, leftSort[leftInd, 3], rightSort[travRightInd, 3])
                height = bottom - top + 1
                # Every time the sum runs, compare computed weights
                # The sum (each operand in the sum was squared) is divided by the area/num of pixels taken
                numZerosPunish = max(np.sqrt(np.count_nonzero(redundancyMatrix[:, leftSort[leftInd, 0]:rightSort[travRightInd, 1] + 1]))
                                     , 1)
                # print(height)
                weight = sum / (np.sqrt((rightSort[travRightInd, 1] - leftSort[leftInd, 0] + 1) * height) + numZerosPunish)
                #print('Count:         ', np.sqrt((rightSort[travRightInd, 1] - leftSort[leftInd, 0] + 1) * len(redundancyMatrix)), numZerosPunish)
                weight /= numZerosPunish
                if weight > retVal['weight']:
                    retVal = {
                        'weight': weight,
                        'left': leftSort[leftInd, 0],
                        'right': rightSort[travRightInd, 1],
                    }
            travRightInd += 1
        leftInd += 1
    assert retVal['right'] <= len(redundancyMatrix[0])
    return retVal['weight'], (retVal['left'], retVal['right'])


def CompareFeatures(first, second):
    winner = {
        'weight': -1,
        'firstLeft': -1,
        'firstRight': -1,
        'secondLeft': -1,
        'secondRight': -1,
        'firstFeatures': None,
        'secondFeatures': None,
        'redundancyMatrix': None,
        'redundancyOffset': -1,
    }
    assert len(first) == len(second)
    smallerFeature, largerFeature, firstIsSmaller = (first, second, True) if len(first[0]) < len(second[0]) \
                                                                            else (second, first, False)
    if len(smallerFeature[0]) < THRESHOLD2:
        return winner
    smallerStart, smallerStop = len(smallerFeature[0]) - THRESHOLD2, len(smallerFeature[0])
    largerStart, largerStop = 0, THRESHOLD2
    # print(len(first[0]), len(second[0]), len(first[0]) + len(second[0]) - 2*(THRESHOLD2-1) - 1)
    for _ in range(len(first[0]) + len(second[0]) - 2*(THRESHOLD2-1) - 1):
        smallerWindow = smallerFeature[:, smallerStart:smallerStop, :]
        largerWindow = largerFeature[:, largerStart:largerStop, :]
        assert smallerStop-smallerStart == largerStop-largerStart, (smallerStart, smallerStop, largerStart, largerStop)
        assert len(first) == len(second) == len(smallerWindow) == len(largerWindow)
        # redundancyMatrix holds the data whether or not a specific element passes the thresholds
        #    when comparing features
        # Note: len(first) is just height which should not change: len(first) == len(second) == len(smallerWindow)...
        assert smallerWindow.shape == largerWindow.shape, (smallerWindow.shape, largerWindow.shape)
        redundancyMatrix = TestCompatibilityOfTwoFeatures(smallerWindow, largerWindow)
        if np.count_nonzero(redundancyMatrix) > redundancyMatrix.size * 0.60:
            weight, redundancyOffset = GetWeight(redundancyMatrix)
            if weight > winner['weight']:
                if firstIsSmaller:
                    winner = {
                        'weight': weight,
                        'firstLeft': smallerStart,
                        'firstRight': smallerStop - 1,
                        'firstFeatures': smallerWindow,
                        'secondLeft': largerStart,
                        'secondRight': largerStop - 1,
                        'secondFeatures': largerWindow,
                        'redundancyMatrix': redundancyMatrix,
                        'redundancyOffset': redundancyOffset,
                    }
                else:
                    winner = {
                        'weight': weight,
                        'firstLeft': largerStart,
                        'firstRight': largerStop - 1,
                        'firstFeatures': largerWindow,
                        'secondLeft': smallerStart,
                        'secondRight': smallerStop - 1,
                        'secondFeatures': smallerWindow,
                        'redundancyMatrix': redundancyMatrix,
                        'redundancyOffset': redundancyOffset,
                    }
                # return winner
        # largerStart increases depending on the value of smallerStart from the previous iteration
        largerStart = 0 if smallerStart != 0 else largerStart + 1
        # smallerStop decreases depending on the value of largerStop from the previous iteration
        smallerStop = len(smallerFeature[0]) if largerStop != len(largerFeature[0]) else smallerStop - 1
        # largerStop keeps increasing towards len(largerFeature)
        largerStop = min(largerStop + 1, len(largerFeature[0]))
        # smallerStart keeps increasing till it reaches the first index of smallerFeature
        smallerStart = max(smallerStart - 1, 0)
        assert smallerStop-smallerStart == largerStop-largerStart, (smallerStart, smallerStop, largerStart, largerStop)
    assert len(largerFeature[0]) - largerStart == THRESHOLD2-1, 'Underlooped' if len(largerFeature[0]) - largerStart > 4 else 'Overlooped'
    return winner


def FitWithinThresh(first, second):
    top1, bottom1 = first[2], first[3]
    top2, bottom2 = second[2], second[3]
    smallerTop, smallerBottom = (top1, bottom1) if bottom1-top1 < bottom2-top2 else (top2, bottom2)
    # If intersection (height) of both CC >= MINHEIGHTPROP * height of the smaller CC, return True
    # If min(bottom1, bottom2) - max(top1, top2) + 1 >= MINHEIGHTPROP * (smallerBottom - smallerTop + 1)
    # Change max(top1, top2) because a negative value is an underflow for unsigned (numpy)
    return min(bottom1, bottom2) + 1 >= MINHEIGHTPROP * (smallerBottom - smallerTop + 1) + max(top1, top2)


def GetDegreeOfRedundancy(pastCCFeatures, currentCCFeatures, pastImg, currentImg):
    winner = {
        'weight': -1,
        'currentLeft': -1,
        'currentRight': -1,
        'currentTop': -1,
        'currentBottom': -1,
        'currentFeatures': None,
        'pastLeft': -1,
        'pastRight': -1,
        'pastTop': -1,
        'pastBottom': -1,
        'redundancyMatrix': None,
        'pastFeatures': None,
        'redundancyOffset': -1,
    }
    # Compare the height of both CC and assign respectively
    smallerCCFeatures, largerCCFeatures, currentIsSmaller = (currentCCFeatures, pastCCFeatures, True) \
                                                                if currentCCFeatures['info'][8] < pastCCFeatures['info'][8]\
                                                                    else (pastCCFeatures, currentCCFeatures, False)
    # From the outermost dimension: row, column, the 10 feature + 2 metafeature
    # "Smaller" and "Larger" is based on which CC has a "smaller" and "larger" height
    smallerFeatures = smallerCCFeatures['features']
    largerFeatures = largerCCFeatures['features']
    # changeTop holds the offset of the top of the CC with a smaller height
    for changeTop in range(-ROW_WINDOW_SIZE, ROW_WINDOW_SIZE + 1):
        nonOverlapSize = ROW_WINDOW_SIZE - ROW_OVERLAP_SIZE
        # height = nonOverlapSize * x + ROW_WINDOW_SIZE (for some x)
        smallerTop = max(0 , int(np.ceil((smallerCCFeatures['info'][2] + 1 - ROW_WINDOW_SIZE) / nonOverlapSize)) + changeTop)
        largerTop = max(0, int(np.ceil((largerCCFeatures['info'][2] + 1 - ROW_WINDOW_SIZE) / nonOverlapSize)))
        # -1 for consistency where bottom holds the last index
        smallerBottom, largerBottom = smallerTop + len(smallerFeatures) - 1, largerTop + len(largerFeatures) - 1
        # "Greater" and "Lesser" are based on the respective values of both top (same for bottom)
        greaterTop = max(smallerTop, largerTop)
        lesserBottom = min(smallerBottom, largerBottom)
        if greaterTop >= lesserBottom:
            continue
        # Crop the feature matrix so that the top and bottom of both CC features exactly intersect
        smallerCropped = smallerFeatures[greaterTop-smallerTop:lesserBottom-smallerTop+1]
        largerCropped = largerFeatures[greaterTop-largerTop:lesserBottom-largerTop+1]
        assert greaterTop-smallerTop == 0 or greaterTop-largerTop == 0
        assert greaterTop-smallerTop < lesserBottom-smallerTop+1
        assert greaterTop-largerTop < lesserBottom-largerTop+1
        assert len(smallerCropped) == len(largerCropped), (smallerCropped.shape, largerCropped.shape)
        comparison = CompareFeatures(smallerCropped, largerCropped)
        # if currentCCFeatures['index'] == 1:
        #     cv.imshow('past', testing.imshow_components(pastCCFeatures['img']))
        #     cv.imshow('current', testing.imshow_components(currentCCFeatures['img']))
        #     cv.waitKey()
        #     cv.destroyAllWindows()
        # if comparison['weight'] > 0:
        #     if currentIsSmaller:
        #         cv.imshow('current', testing.imshow_components(
        #             currentCCFeatures['img'][greaterTop - smallerTop:
        #                                      greaterTop - smallerTop + currentCCFeatures['info'][8],
        #                                         comparison['firstLeft']:
        #                                         comparison['firstRight']]))
        #         cv.imshow('past', testing.imshow_components(
        #             pastCCFeatures['img'][greaterTop - largerTop:
        #                                   greaterTop - largerTop + pastCCFeatures['info'][8],
        #                                     comparison['secondLeft']:
        #                                     comparison['secondRight']]))
        #     else:
        #         cv.imshow('current', testing.imshow_components(
        #             pastCCFeatures['img'][greaterTop - largerTop:
        #                                   greaterTop - largerTop + pastCCFeatures['info'][8],
        #                                     comparison['secondLeft']:
        #                                     comparison['secondRight']]))
        #         cv.imshow('past', testing.imshow_components(
        #             currentCCFeatures['img'][greaterTop - smallerTop:
        #                                      greaterTop - smallerTop + currentCCFeatures['info'][8],
        #                                         comparison['firstLeft']:
        #                                         comparison['firstRight']]))
        # cv.waitKey()
        # cv.destroyAllWindows()
        if comparison['weight'] > winner['weight']:
            assert comparison['firstLeft'] == 0 or comparison['secondLeft'] == 0
            assert greaterTop - largerTop == 0 or greaterTop - smallerTop == 0
            assert comparison['firstLeft'] <= comparison['firstLeft'] + comparison['redundancyOffset'][0] <= \
                   comparison['firstLeft'] + comparison['redundancyOffset'][1] <= comparison['firstRight'], (
                    comparison['firstLeft'], comparison['firstLeft'] + comparison['redundancyOffset'][0], \
                    comparison['firstLeft'] + comparison['redundancyOffset'][1], comparison['firstRight']
            )
            assert comparison['secondLeft'] <= comparison['secondLeft'] + comparison['redundancyOffset'][0] <= \
                   comparison['secondLeft'] + comparison['redundancyOffset'][1] <= comparison['secondRight'], (
                    comparison['secondLeft'], comparison['secondLeft'] + comparison['redundancyOffset'][0], \
                    comparison['secondLeft'] + comparison['redundancyOffset'][1], comparison['secondRight']
            )
            if currentIsSmaller:
                winner = {
                    'weight': comparison['weight'],
                    'currentLeft': comparison['firstLeft'],
                    'currentRight': comparison['firstRight'],
                    'currentTop': greaterTop - smallerTop,
                    'currentBottom': lesserBottom - smallerTop,
                    'currentFeatures': comparison['firstFeatures'],
                    'pastLeft': comparison['secondLeft'],
                    'pastRight': comparison['secondRight'],
                    'pastTop': greaterTop - largerTop,
                    'pastBottom': lesserBottom - largerTop,
                    'pastFeatures': comparison['secondFeatures'],
                    'redundancyMatrix': comparison['redundancyMatrix'],
                    'redundancyOffset': comparison['redundancyOffset'],
                }
            else:
                winner = {
                    'weight': comparison['weight'],
                    'pastLeft': comparison['firstLeft'],
                    'pastRight': comparison['firstRight'],
                    'pastTop': greaterTop - smallerTop,
                    'pastBottom': lesserBottom - smallerTop,
                    'pastFeatures': comparison['firstFeatures'],
                    'currentLeft': comparison['secondLeft'],
                    'currentRight': comparison['secondRight'],
                    'currentTop': greaterTop - largerTop,
                    'currentBottom': lesserBottom - largerTop,
                    'currentFeatures': comparison['secondFeatures'],
                    'redundancyMatrix': comparison['redundancyMatrix'],
                    'redundancyOffset': comparison['redundancyOffset'],
                }
    return winner


def RecreateImgFromFeatures(redundancyMatrix, img, topLeft, bottomRight):
    left, right, top, bottom = topLeft[0], bottomRight[0], topLeft[1], bottomRight[1]
    width, height = right - left + 1, bottom - top + 1
    mask = redundancyMatrix.copy()
    mask[mask==1] = 255
    mask = cv.resize(mask, dsize=(width, height), interpolation=cv.INTER_LINEAR)
    mask[mask>0] = 1
    subImg = img[top:bottom+1, left:right+1]
    assert subImg.shape == mask.shape, (subImg.shape, mask.shape)
    retImg = cv.bitwise_and(subImg, subImg, mask=mask)
    return retImg


def GetMergedType(winner, firstWidth, secondWidth):
    firstBox, secondBox = (winner['currentLeft'], winner['currentRight']), (winner['pastLeft'], winner['pastRight'])
    # lesserLeft, greaterLeft = (firstBox[0], secondBox[0]) if firstBox[0] < secondBox[0] else (secondBox[0], firstBox[0])
    # lesserRight, greaterRight = (firstBox[1], secondBox[1]) if firstBox[1] < secondBox[1] else (secondBox[1], firstBox[1])

    tipFirstLen, tipSecondLen = np.ceil(firstWidth * THRESHOLD3), np.ceil(secondWidth * THRESHOLD3)
    firstOffsetStart = firstBox[0] + winner['redundancyOffset'][0]
    firstOffsetStop = firstBox[0] + winner['redundancyOffset'][1]
    secondOffsetStart = secondBox[0] + winner['redundancyOffset'][0]
    secondOffsetStop = secondBox[0] + winner['redundancyOffset'][1]
    if firstBox[1] - firstBox[0] + 1 == secondWidth or secondBox[1] - secondBox[0] + 1 == firstWidth:
        # One is the subset of the other
        return 1
    elif tipFirstLen - firstOffsetStart > (firstOffsetStop - firstOffsetStart + 1) * THRESHOLD3 and \
            (secondOffsetStop + 1) - (secondWidth - tipSecondLen) > (secondOffsetStop - secondOffsetStart + 1) * THRESHOLD3:
        return 1
    elif tipSecondLen - secondOffsetStart > (secondOffsetStop - secondOffsetStart + 1) * THRESHOLD3 and \
            (firstOffsetStop + 1) - (firstWidth - tipFirstLen) > (firstOffsetStop - firstOffsetStart + 1) * THRESHOLD3:
        return 1
    else:
        return 2


def CreateMergedFeatures(relative, intersection, final, height, width):
    finalX = relative[0] + intersection[0] - final[0]
    finalY = relative[1] + intersection[1] - final[1]
    return finalX, finalX + width - 1, finalY, finalY + height - 1


def CheckBlackIntersection(firstFeatures, firstBox, secondFeatures, secondBox):
    greaterTop = max(firstBox[2], secondBox[2])
    lesserBottom = min(firstBox[2] + len(firstFeatures), secondBox[2] + len(secondFeatures))
    greaterLeft = max(firstBox[0], secondBox[0])
    lesserRight = min(firstBox[0] + len(firstFeatures[0]), secondBox[0] + len(secondFeatures[0]))
    assert greaterTop <= lesserBottom and greaterLeft <= lesserRight, (firstBox, secondBox, firstFeatures.shape, secondFeatures.shape)
    currentIntersect = [greaterLeft - firstBox[0], lesserRight - firstBox[0],
                        greaterTop - firstBox[2], lesserBottom - firstBox[2]]
    pastIntersect = [greaterLeft - secondBox[0], lesserRight - secondBox[0],
                        greaterTop - secondBox[2], lesserBottom - secondBox[2]]
    currentMerged = firstFeatures[currentIntersect[2]:currentIntersect[3]+1, currentIntersect[0]:currentIntersect[1]+1]
    pastMerged = secondFeatures[pastIntersect[2]:pastIntersect[3]+1, pastIntersect[0]:pastIntersect[1]+1]
    # If even one in currentMerged or pastMerged has a non-blank feature, return False else return True
    # If currentMerged has non-blank features or pastMerged has non-blank features, return False else return True
    return not np.logical_or(np.any(currentMerged[:, :, 8]!=0), np.any(pastMerged[:, :, 8]!=0))


def TryToMergeFeatures(imgsFeatures, currentImgFeatures, matchedLabels, currentMergedFeatures, pastMergedFeatures, currentMatchedLabelsAssoc, currentRelative,
                       pastRelative, currentImgNum, currentImgInd, pastImgNum, pastImgInd, maxWeight, winner, matched):
    assert matched in [0, 1, 2]
    currentCoord = matchedLabels[currentImgNum, currentImgInd]
    pastCoord = matchedLabels[pastImgNum, pastImgInd]
    weight, currentImgInd, pastImgNum, pastImgInd, winner, currentCCFeatures, pastCCFeatures = maxWeight
    if matched == 0:
        # Check coordinate of current
        assert currentCoord != -1 and pastCoord == -1 and currentMatchedLabelsAssoc[currentImgInd] != -1 \
               and np.all(currentRelative != -10**4) and np.all(pastRelative == -10**4), (currentCoord, pastCoord, currentMatchedLabelsAssoc[currentImgInd],
                                                                                  currentRelative, pastRelative)
        coord = pastMergedFeatures[currentMatchedLabelsAssoc[currentImgInd]]
        merged = list(CreateMergedFeatures(currentRelative, (winner['currentLeft'], winner['currentTop']),
                                      (winner['pastLeft'], winner['pastTop']),
                                      len(pastCCFeatures['features']), len(pastCCFeatures['features'][0])))
        for box in coord[:4]:
            if box[4] != currentImgNum or box[5] != currentImgInd:
                boxNum, boxInd = box[4], currentImgFeatures[box[5]]['index'] if box[4] == currentImgNum else box[5]
                greaterLeft = max(merged[0], box[0])
                lesserRight = min(merged[1], box[1])
                greaterTop = max(merged[2], box[2])
                lesserBottom = min(merged[3], box[3])
                if (greaterLeft <= lesserRight and greaterTop <= lesserBottom) \
                        and not CheckBlackIntersection(imgsFeatures[boxNum][boxInd]['features'], box, pastCCFeatures['features'], merged):
                    return False
        for box in currentMergedFeatures[currentCoord]:
            greaterLeft = max(winner['currentLeft'], box[0])
            lesserRight = min(winner['currentRight'], box[1])
            greaterTop = max(winner['currentTop'], box[2])
            lesserBottom = min(winner['currentBottom'], box[3])
            if greaterLeft <= lesserRight and greaterTop <= lesserBottom:
                return False
    elif matched == 1:
        # Check coordinate of past
        assert currentCoord == -1 and pastCoord != -1 and np.all(currentRelative == -10**4) and np.all(pastRelative != -10**4) \
               and currentMatchedLabelsAssoc[currentImgInd] == -1
        merged = list(CreateMergedFeatures(pastRelative, (winner['pastLeft'], winner['pastTop']),
                                      (winner['currentLeft'], winner['currentTop']),
                                      len(currentCCFeatures['features']), len(currentCCFeatures['features'][0])))
        coord = pastMergedFeatures[pastCoord]
        for box in coord[:4]:
            if box[4] != pastImgNum or box[5] != pastImgInd:
                boxNum, boxInd = box[4], currentImgFeatures[box[5]]['index'] if box[4] == currentImgNum else box[5]
                greaterLeft = max(merged[0], box[0])
                lesserRight = min(merged[1], box[1])
                greaterTop = max(merged[2], box[2])
                lesserBottom = min(merged[3], box[3])
                if (greaterLeft <= lesserRight and greaterTop <= lesserBottom) \
                       and not CheckBlackIntersection(imgsFeatures[boxNum][boxInd]['features'], box, currentCCFeatures['features'], merged):
                    return False
    else:
        # Check both coordinates
        assert currentCoord != -1 and pastCoord != -1 and np.all(currentRelative != -10**4) and np.all(pastRelative != -10**4) \
               and currentMatchedLabelsAssoc[currentImgInd] != -1
        merged = list(CreateMergedFeatures(currentRelative, (winner['currentLeft'], winner['currentTop']),
                                      (winner['pastLeft'], winner['pastTop']),
                                      len(pastCCFeatures['features']), len(pastCCFeatures['features'][0])))
        oldOrigin = [merged[0] - pastRelative[0], merged[1] - pastRelative[1]]
        pastCurrentCoord = pastMergedFeatures[currentMatchedLabelsAssoc[currentImgInd]]
        pastPastCoord = pastMergedFeatures[pastCoord]
        for pastBox in pastPastCoord[:4]:
            pastBoxNum, pastBoxInd = pastBox[4], currentImgFeatures[pastBox[5]]['index'] if pastBox[4] == currentImgNum else pastBox[5]
            for currentBox in pastCurrentCoord[:4]:
                currentBoxNum, currentImgInd = currentBox[4], currentImgFeatures[currentBox[5]]['index'] \
                                                                    if currentBox[4] == currentImgNum else currentBox[5]
                if currentBox[4] != pastImgNum or currentBox[5] != pastImgInd:
                    newPastBox = [pastBox[0] + oldOrigin[0],
                              pastBox[1] + oldOrigin[0],
                              pastBox[2] + oldOrigin[1],
                              pastBox[3] + oldOrigin[1]]
                    greaterLeft = max(newPastBox[0], currentBox[0])
                    lesserRight = min(newPastBox[1], currentBox[1])
                    greaterTop = max(newPastBox[2], currentBox[2])
                    lesserBottom = min(newPastBox[3], currentBox[3])
                    if (greaterLeft <= lesserRight and greaterTop <= lesserBottom) \
                           and not CheckBlackIntersection(imgsFeatures[currentBoxNum][currentImgInd]['features'], currentBox,
                                                          imgsFeatures[pastBoxNum][pastBoxInd]['features'], newPastBox):
                        return False
        for box in currentMergedFeatures[currentCoord]:
            greaterLeft = max(winner['currentLeft'], box[0])
            lesserRight = min(winner['currentRight'], box[1])
            greaterTop = max(winner['currentTop'], box[2])
            lesserBottom = min(winner['currentBottom'], box[3])
            if greaterLeft <= lesserRight and greaterTop <= lesserBottom:
                return False
    return True


def PositionRedundancy(intersect, CCFeatures, intersectHeight, intersectWidth):
    nonOverlapSize = COL_WINDOW_SIZE - COL_OVERLAP_SIZE
    intersectLeft = intersect[0] * nonOverlapSize + COL_WINDOW_SIZE
    intersectRight = intersect[1] * nonOverlapSize + COL_WINDOW_SIZE
    nonOverlapSize = ROW_WINDOW_SIZE - ROW_OVERLAP_SIZE
    intersectTop = intersect[2] * nonOverlapSize + ROW_WINDOW_SIZE
    intersectBottom = intersect[3] * nonOverlapSize + ROW_WINDOW_SIZE
    assert CCFeatures['info'][8] == len(CCFeatures['origImg']) and CCFeatures['info'][7] == len(
        CCFeatures['origImg'][0])
    origImgPointers = CCFeatures['origImg'][intersectTop:intersectTop + intersectHeight,
                      intersectLeft:intersectLeft + intersectWidth]
    imgNumInds = origImgPointers[:, :, 0].flatten()
    rowInds = origImgPointers[:, :, 1].flatten()
    colInds = origImgPointers[:, :, 2].flatten()
    return imgNumInds, rowInds, colInds


def RecordRedundancy(currentRedundancyColorer, pastRedundancyColorer, currentCCFeatures, pastCCFeatures, imgNum, intersectCurrent, intersectPast, redundancyOffset):
    global redundancyCounter
    # testing.ShowChosenRedundancy2(currentCCFeatures, pastCCFeatures, {
    #     'currentLeft': intersectCurrent[0],
    #     'currentRight': intersectCurrent[1],
    #     'currentTop': intersectCurrent[2],
    #     'currentBottom': intersectCurrent[3],
    #     'pastLeft': intersectPast[0],
    #     'pastRight': intersectPast[1],
    #     'pastTop': intersectPast[2],
    #     'pastBottom': intersectPast[3],
    # })
    assert intersectCurrent[1] - intersectCurrent[0] == intersectPast[1] - intersectPast[0] and \
           intersectCurrent[3] - intersectCurrent[2] == intersectPast[3] - intersectPast[2]

    nonOverlapSize = ROW_WINDOW_SIZE - ROW_OVERLAP_SIZE
    # Scale from feature to image, and make sure the obtained height does not go beyond the image height from intersectTop
    intersectHeight = min((intersectPast[3] - intersectPast[2] + 1) * nonOverlapSize + ROW_WINDOW_SIZE,
                          pastCCFeatures['info'][8] - (intersectPast[2] * nonOverlapSize + ROW_WINDOW_SIZE),
                          currentCCFeatures['info'][8] - (intersectCurrent[2] * nonOverlapSize + ROW_WINDOW_SIZE))
    nonOverlapSize = COL_WINDOW_SIZE - COL_OVERLAP_SIZE
    # Scale from feature to image, and make sure the obtained width does not go beyond the image width from intersectLeft
    intersectWidth = min((intersectPast[1] - intersectPast[0] + 1) * nonOverlapSize + COL_WINDOW_SIZE,
                         pastCCFeatures['info'][7] - (intersectPast[0] * nonOverlapSize + COL_WINDOW_SIZE),
                         currentCCFeatures['info'][7] - (intersectCurrent[0] * nonOverlapSize + COL_WINDOW_SIZE))

    pastImgInds, pastRowInds, pastColInds = PositionRedundancy(intersectPast, pastCCFeatures,
                                                                intersectHeight, intersectWidth)
    pastLabels = pastRedundancyColorer[pastImgInds, pastRowInds, pastColInds]
    # np.unique is already sorted
    uniqueInds = np.unique(pastImgInds)
    colors = redundancyCounter + np.searchsorted(uniqueInds, pastImgInds)
    pastRedundancyColorer[pastImgInds, pastRowInds, pastColInds] = np.where(pastLabels == 0,
                                                                        colors, pastLabels)
    # redundancyDrawer[pastImgInds, pastRowInds, pastColInds] = np.where(redundancyVals == 255, 1, redundancyVals)

    currentImgInds, currentRowInds, currentColInds = PositionRedundancy(intersectCurrent, currentCCFeatures,
                                                                        intersectHeight, intersectWidth)
    currentLabels = currentRedundancyColorer[currentImgInds, currentRowInds, currentColInds]
    currentRedundancyColorer[currentImgInds, currentRowInds, currentColInds] = np.where(np.logical_and(pastLabels != 0, currentLabels == 0),
                                                                                 pastLabels, currentLabels)
    currentLabels = currentRedundancyColorer[currentImgInds, currentRowInds, currentColInds]
    currentRedundancyColorer[currentImgInds, currentRowInds, currentColInds] = np.where(currentLabels == 0,
                                                                                 colors, currentLabels)
    # redundancyDrawer[currentImgInds, currentRowInds, currentColInds] = np.where(redundancyVals == 255, 1, redundancyVals)
    assert uniqueInds.ndim == 1
    redundancyCounter += uniqueInds.size


def AddNewRedundancy(imgsFeatures, currentImgFeatures, maxWeight, currentMatchedLabelsAssoc, matchedLabels, currentMergedFeatures,
                     pastMergedFeatures, imgNum, type2s, relatives, type2Archives, currentRedundancyColorer, pastRedundancyColorer,
                     matched=-1):
    global countMerged
    weight, currentImgInd, pastImgNum, pastImgInd, winner, currentCCFeatures, pastCCFeatures = maxWeight
    assert imgNum != pastImgNum
    RecordRedundancy(currentRedundancyColorer, pastRedundancyColorer, currentCCFeatures, pastCCFeatures, imgNum,
                        (winner['currentLeft'], winner['currentRight'], winner['currentTop'], winner['currentBottom']),
                        (winner['pastLeft'], winner['pastRight'], winner['pastTop'], winner['pastBottom']),
                                                                            winner['redundancyOffset'])
    assert currentCCFeatures['index'] == currentImgFeatures[currentImgInd]['index']
    assert np.all(np.logical_and(imgsFeatures[imgNum][currentCCFeatures['index']]['img'] == currentCCFeatures['img'],
           currentCCFeatures['img'] == currentImgFeatures[currentImgInd]['img']))
    # if (pastImgNum, pastImgInd, imgNum, currentImgInd) in [(0, 3, 1, 4), (0, 3, 1, 5)] or (pastImgNum, pastImgInd) in [(0, 3)]:
    #     print((pastImgNum, pastImgInd, imgNum, currentImgInd))
    #     print(matched)
    #     testing.ShowChosenRedundancy(currentCCFeatures, pastCCFeatures, winner)
    if type2s[pastImgNum, pastImgInd] is False and \
            GetMergedType(winner, currentCCFeatures['info'][7], pastCCFeatures['info'][7]) == 2:
        type2s[pastImgNum, pastImgInd] = True
        newArchive = pastCCFeatures.copy()
        newArchive['img'] = pastCCFeatures['img'].copy()
        newArchive['info'] = pastCCFeatures['info'].copy()
        newArchive['features'] = pastCCFeatures['features'].copy()
        type2Archives.append(newArchive)
    if matched == -1:
        countMerged[0] += 1
        assert matchedLabels[imgNum, currentImgInd] == -1 and matchedLabels[pastImgNum, pastImgInd] == -1
        assert np.all(relatives[pastImgNum, pastImgInd] == -10**4)
        assert np.all(relatives[imgNum, currentImgInd] == -10**4)
        # The very first relative is at (0, 0) or top left (left, top) of the first merged past
        mergedFeatures = CreateMergedFeatures((0, 0), (winner['pastLeft'], winner['pastTop']),
                                      (winner['currentLeft'], winner['currentTop']),
                                      len(currentCCFeatures['features']), len(currentCCFeatures['features'][0]))
        matchedLabels[imgNum, currentImgInd] = len(currentMergedFeatures)
        currentMergedFeatures.append([])
        currentMergedFeatures[-1].append([winner['currentLeft'], winner['currentRight'], winner['currentTop'], winner['currentBottom']])
        relatives[imgNum, currentImgInd] = [mergedFeatures[0], mergedFeatures[2]]
        assert currentMatchedLabelsAssoc[currentImgInd] == -1

        currentMatchedLabelsAssoc[currentImgInd] = len(pastMergedFeatures)
        matchedLabels[pastImgNum, pastImgInd] = len(pastMergedFeatures)
        relatives[pastImgNum, pastImgInd] = [0, 0]
        pastMergedFeatures.append([])
        # Other than the left, right, top, bottom coordinates now being occupied because of merging given by mergedFeatures,
        #    we also include a "pointer" represented by currentCCType, imgNum, and currentLabelNum to point back to the
        #    information (info, features, img, etc.) of the newly merged feature.
        pastMergedFeatures[-1].append(np.array([0, len(pastCCFeatures['features'][0]) - 1, 0, len(pastCCFeatures['features']) - 1
                                                   , pastImgNum, pastImgInd]))
        pastMergedFeatures[-1].append(np.r_[mergedFeatures, [imgNum, currentImgInd]])
    elif matched == 0:
        countMerged[1] += 1
        # Repeated the current CC for a different past CC
        assert matchedLabels[imgNum, currentImgInd] != -1 and matchedLabels[pastImgNum, pastImgInd] == -1
        assert np.all(relatives[imgNum, currentImgInd] != -10**4)
        assert np.all(relatives[pastImgNum, pastImgInd] == -10**4)
        # Relative is now the current since the current already has values in relatives,
        #    the origin is therefore the very first past that was merged with current and
        #    we base this (current) past to those values via its relation to current
        mergedFeatures = CreateMergedFeatures(relatives[imgNum, currentImgInd], (winner['currentLeft'], winner['currentTop']),
                                      (winner['pastLeft'], winner['pastLeft']),
                                      len(pastCCFeatures['features']), len(pastCCFeatures['features'][0]))
        currentMergedFeatures[matchedLabels[imgNum, currentImgInd]].append(
            [winner['currentLeft'], winner['currentRight'], winner['currentTop'], winner['currentBottom']])

        assert currentMatchedLabelsAssoc[currentImgInd] != -1
        matchedLabels[pastImgNum, pastImgInd] = currentMatchedLabelsAssoc[currentImgInd]
        relatives[pastImgNum, pastImgInd] = [mergedFeatures[0], mergedFeatures[2]]
        pastMergedFeatures[currentMatchedLabelsAssoc[currentImgInd]].append(
                                            np.r_[mergedFeatures, [pastImgNum, pastImgInd]])
    elif matched == 1:
        countMerged[2] += 1
        # Repeated the past CC for a different current CC
        assert matchedLabels[imgNum, currentImgInd] == -1 and matchedLabels[pastImgNum, pastImgInd] != -1
        assert np.all(relatives[imgNum, currentImgInd] == -10**4)
        assert np.all(relatives[pastImgNum, pastImgInd] != -10**4)
        # Past got repeated so theres a possibility the origin is not in the top left of the (current) past since
        #    the previous merging with this past may be when matched == 0 making the origin be another/different past
        mergedFeatures = CreateMergedFeatures(relatives[pastImgNum, pastImgInd], (winner['pastLeft'], winner['pastTop']),
                                      (winner['currentLeft'], winner['currentTop']),
                                      len(currentCCFeatures['features']), len(currentCCFeatures['features'][0]))
        matchedLabels[imgNum, currentImgInd] = len(currentMergedFeatures)
        currentMergedFeatures.append([])
        currentMergedFeatures[-1].append([winner['currentLeft'], winner['currentRight'], winner['currentTop'], winner['currentBottom']])
        relatives[imgNum, currentImgInd] = [mergedFeatures[0], mergedFeatures[2]]
        assert currentMatchedLabelsAssoc[currentImgInd] == -1
        currentMatchedLabelsAssoc[currentImgInd] = matchedLabels[pastImgNum, pastImgInd]
        pastMergedFeatures[currentMatchedLabelsAssoc[currentImgInd]].append(
                                            np.r_[mergedFeatures, [imgNum, currentImgInd]])
    else:
        countMerged[3] += 1
        # The current and past already have their existing coordinates.
        # We choose to transfer all boxes in the past coordinate to the current based on their intersection
        assert matchedLabels[imgNum, currentImgInd] != matchedLabels[pastImgNum, pastImgInd]
        assert np.all(relatives[imgNum, currentImgInd] != -10**4)
        assert np.all(relatives[pastImgNum, pastImgInd] != -10**4)
        mergedFeatures = CreateMergedFeatures(relatives[imgNum, currentImgInd], (winner['currentLeft'], winner['currentTop']),
                                      (winner['pastLeft'], winner['pastLeft']),
                                      len(pastCCFeatures['features']), len(pastCCFeatures['features'][0]))
        currentMergedFeatures[matchedLabels[imgNum, currentImgInd]].append(
            (winner['currentLeft'], winner['currentRight'], winner['currentTop'], winner['currentBottom']) )

        # oldOrigin holds the coordinates of the origin of the past based on the (left, top) coordinates of the past
        #    in the current coordinate system (from mergedFeatures) and the coordinates of the past in the past coordinate
        #    system (from relatives)
        oldOrigin = [mergedFeatures[0] - relatives[pastImgNum, pastImgInd, 0],
                     mergedFeatures[2] - relatives[pastImgNum, pastImgInd, 1]]
        # oldOrigin = mergedFeatures[[0, 2]] - relatives[pastImgNum, pastImgInd]
        for box in pastMergedFeatures[matchedLabels[pastImgNum, pastImgInd]]:
            left, right, top, bottom = box[:4]
            left, right, top, bottom = left+oldOrigin[0], right+oldOrigin[0], top+oldOrigin[1], bottom+oldOrigin[1]
            pastMergedFeatures[currentMatchedLabelsAssoc[currentImgInd]].append(
                                                np.array([left, right, top, bottom, box[4], box[5]]))
            assert matchedLabels[box[4], box[5]] != -1
            assert np.all(relatives[box[4], box[5]] != -10**4)
            relatives[box[4], box[5]] = [left, top]
            if imgNum == box[4]:
                assert currentMatchedLabelsAssoc[box[5]] != -1
                currentMatchedLabelsAssoc[box[5]] = currentMatchedLabelsAssoc[currentImgInd]
            else:
                matchedLabels[box[4], box[5]] = currentMatchedLabelsAssoc[currentImgInd]
        assert matchedLabels[pastImgNum, pastImgInd] == currentMatchedLabelsAssoc[currentImgInd]


def OverwriteImages(imgsFeatures, imgNum, currentMergedFeatures, pastMergedFeatures,
                    matchedLabels, deleteInds, imgsPhraseLabels, imgsNonTextLabels, currentImgFeatures):
    scaleX = COL_WINDOW_SIZE - COL_OVERLAP_SIZE
    scaleY = ROW_WINDOW_SIZE - ROW_OVERLAP_SIZE
    # For assertion that when a CC was used, it wont be used again
    maxLengthPossible = max([len(imgsFeatures[i]) for i in range(imgNum + 1)])
    usedUpCCs = np.zeros((imgNum + 1, maxLengthPossible), dtype=bool)
    if len(pastMergedFeatures) > 0:
        # Since some pastMergedFeatures were combined with others, some indices should not be used
        included = np.unique(matchedLabels[:imgNum, :])
        included = included[included >= 0]
        assert len(included) > 0
        # firstAppearances holds the image number and index of the least index in the least imgNum
        # Initialize firstAppearances with the first box in every coordinate
        firstAppearances = np.zeros((np.max(included) + 1, 6), dtype=np.int16)
        for ind in included:
            firstAppearances[ind] = pastMergedFeatures[ind][0]
        for ind in included:
            # edges helps know how much img increased in size (box of img) after merging with redundancy
            box = pastMergedFeatures[ind][0]
            img = imgsFeatures[box[4]][box[5]]['img']
            edges = [box[0] * scaleX,
                     box[0] * scaleX + len(img[0]),
                     box[2] * scaleY,
                     box[2] * scaleY + len(img)]
            coordinate = pastMergedFeatures[ind]
            nonTextExists = False
            for box in coordinate:
                boxNum, boxInd = box[4], box[5]
                if boxNum == imgNum:
                    boxInd = currentImgFeatures[box[5]]['index']
                if imgsFeatures[boxNum][boxInd]['type'] == 'nonText':
                    nonTextExists = True
                if boxNum < firstAppearances[ind, 4] or (boxNum == firstAppearances[ind, 4] and boxInd < firstAppearances[ind, 5]):
                    firstAppearances[ind] = box
                box[:2] = box[:2] * scaleX
                box[2:4] = box[2:4] * scaleY
                img = imgsFeatures[boxNum][boxInd]['img']
                edges[0], edges[1] = min(edges[0], box[0]), max(edges[1], box[0] + len(img[0]))
                edges[2], edges[3] = min(edges[2], box[2]), max(edges[3], box[2] + len(img))
            newImg = np.zeros((edges[3] - edges[2] + 1, edges[1] - edges[0] + 1), dtype=np.uint8)
            newOrigImg = np.zeros((*newImg.shape, 3), dtype=np.uint16)
            # Reverse the vector that had gone from old coord to (0, 0) of newImg
            edges = [-edge for edge in edges]
            for box in coordinate:
                if box[4] != imgNum:
                    boxNum, boxInd = box[4], box[5]
                    # Place the past CCs first in the newImg before the CC in the current imgNum
                    img = imgsFeatures[boxNum][boxInd]['img']
                    origImg = imgsFeatures[boxNum][boxInd]['origImg']
                    assert len(img) == len(origImg) and len(img[0]) == len(origImg[0])
                    newImg[edges[2]+box[2]:edges[2]+box[2]+len(img), edges[0]+box[0]:edges[0]+box[0]+len(img[0])] = img
                    newOrigImg[edges[2]+box[2]:edges[2]+box[2]+len(origImg), edges[0]+box[0]:edges[0]+box[0]+len(origImg[0])] = origImg
                    assert len(img) == len(origImg) and len(img[0]) == len(origImg[0])
                    deleteInds[boxNum, boxInd] = True
                    assert usedUpCCs[boxNum, boxInd] == False
                    usedUpCCs[boxNum, boxInd] = True
            for box in coordinate:
                if box[4] == imgNum:
                    boxNum, boxInd = imgNum, currentImgFeatures[box[5]]['index']
                    # Place the current CCs now
                    img = imgsFeatures[boxNum][boxInd]['img']
                    origImg = imgsFeatures[boxNum][boxInd]['origImg']
                    assert len(img) == len(origImg) and len(img[0]) == len(origImg[0])
                    newImg[edges[2]+box[2]:edges[2]+box[2]+len(img), edges[0]+box[0]:edges[0]+box[0]+len(img[0])] = img
                    newOrigImg[edges[2]+box[2]:edges[2]+box[2]+len(origImg), edges[0]+box[0]:edges[0]+box[0]+len(origImg[0])] = origImg
                    assert len(img) == len(origImg) and len(img[0]) == len(origImg[0])
                    deleteInds[boxNum, boxInd] = True
                    assert usedUpCCs[boxNum, boxInd] == False
                    usedUpCCs[boxNum, boxInd] = True
            info = Step5.GetLabelsInfo(np.where(newImg!=0, 1, 0))[0]
            # If the edge rows/columns still have blanks, delete those rows/columns
            newImg = newImg[info[2]:info[3]+1, info[0]:info[1]+1]
            newOrigImg = newOrigImg[info[2]:info[3]+1, info[0]:info[1]+1]
            assert np.all(firstAppearances[:, 4] != imgNum)
            imgsFeatures[firstAppearances[ind, 4]][firstAppearances[ind, 5]]['img'] = newImg
            imgsFeatures[firstAppearances[ind, 4]][firstAppearances[ind, 5]]['origImg'] = newOrigImg
            # firstAppearance points to the box so after box was scaled it should have also affected records in firstAppearances
            firstAppearLeft, firstAppearTop = firstAppearances[ind, 0] + edges[0], firstAppearances[ind, 2] + edges[2]
            prevLeftTop = imgsFeatures[firstAppearances[ind, 4]][firstAppearances[ind, 5]]['info'][[0, 2]]
            newLeft, newTop = prevLeftTop[0] - firstAppearLeft, prevLeftTop[1] - firstAppearTop

            newBottom = newTop + len(newImg) - 1
            newRight = newLeft + len(newImg[0]) - 1
            # Top and bottom seeds now inconsistent
            imgsFeatures[firstAppearances[ind, 4]][firstAppearances[ind, 5]]['info'] = np.r_[[newLeft, newRight, newTop, newBottom], info[4:]]
            imgsFeatures[firstAppearances[ind, 4]][firstAppearances[ind, 5]]['top'] = newTop
            imgsFeatures[firstAppearances[ind, 4]][firstAppearances[ind, 5]]['minHeight'] = newTop - ((1 - MINHEIGHTPROP) * info[9] // info[7])
            imgsFeatures[firstAppearances[ind, 4]][firstAppearances[ind, 5]]['features'] = Step6.main(np.where(newImg!=0, 1, 0), 1)[0]
            deleteInds[firstAppearances[ind, 4], firstAppearances[ind, 5]] = False
            if nonTextExists:
                imgsFeatures[firstAppearances[ind, 4]][firstAppearances[ind, 5]]['type'] = 'nonText'
    newImgsFeatures = [[imgsFeatures[i][j] for j in range(len(imgsFeatures[i])) if not deleteInds[i, j]]
                                    for i in range(imgNum + 1)]
    # keepCurrent = [True] * len(currentImgFeatures)
    # for ind, willDelete in enumerate(deleteInds[imgNum]):
    #     if willDelete:
    #         keepCurrent[currentImgFeatures[ind]['index']] = False
    # newImgsFeatures.append(
    #     [imgsFeatures[imgNum][ind] for ind, willKeep in enumerate(keepCurrent) if willKeep])
    for imgFeatures in imgsFeatures[imgNum + 1:]:
        newImgsFeatures.append(imgFeatures)
    imgsFeatures = newImgsFeatures
    return imgsFeatures


def UpdateFeatureInfo(imgsFeatures, redundantHeap, imgNum, imgsPhraseLabels, imgsNonTextLabels,
                      type2Archives, currentRedundancyColorer, pastRedundancyColorer, currentImgFeatures, THRESHOLD1):
    currentMergedFeatures = []
    pastMergedFeatures = []
    maxLengthPossible = max([len(imgsFeatures[i]) for i in range(imgNum + 1)])
    matchedLabels = np.full((imgNum + 1, maxLengthPossible), -1, dtype=np.int16)
    deleteInds = np.zeros((imgNum + 1, maxLengthPossible), dtype=bool)
    relatives = np.full((imgNum + 1, maxLengthPossible, 2), -10**4, dtype=np.int16)
    type2s = np.zeros((imgNum, maxLengthPossible), dtype=bool)
    currentMatchedLabelsAssoc = np.full(len(imgsFeatures[imgNum]), -1, dtype=np.int16)
    while len(redundantHeap) > 0:
        maxWeight = heapq.heappop(redundantHeap)
        weight, currentImgInd, pastImgNum, pastImgInd, winner, currentCCFeatures, pastCCFeatures = maxWeight
        if -weight < THRESHOLD1:
            break
        hasMatched = [matchedLabels[imgNum, currentImgInd] != -1,
                        matchedLabels[pastImgNum, pastImgInd] != -1]
        # testing.ShowChosenRedundancy(currentCCFeatures, pastCCFeatures, winner, COL_WINDOW_SIZE, COL_OVERLAP_SIZE)
        if True in hasMatched:
            matchedInds = hasMatched.index(True)
            if matchedInds == 0 and hasMatched[1]:
                matchedInds = 2
            # Either the current or the past already matched before which may overlap in the redundancy (e.g. 2 currents
            #    having a redundancy on the same feature from a past)
            assert len(currentMergedFeatures)>0 and len(pastMergedFeatures)>0, (len(currentMergedFeatures)>0, len(pastMergedFeatures)==0)
            mergeable = TryToMergeFeatures(imgsFeatures, currentImgFeatures, matchedLabels, currentMergedFeatures, pastMergedFeatures, currentMatchedLabelsAssoc,
                                            relatives[imgNum, currentImgInd], relatives[pastImgNum, pastImgInd],
                                           imgNum, currentImgInd, pastImgNum, pastImgInd, maxWeight, winner, matched=matchedInds)
            # if imgNum == 1 and currentImgInd == 1:
            #     print(mergeable, matchedInds)
            #     testing.ShowChosenRedundancy(currentCCFeatures, pastCCFeatures, winner, plotIt=True)
            if mergeable:
                AddNewRedundancy(imgsFeatures, currentImgFeatures, maxWeight, currentMatchedLabelsAssoc, matchedLabels, currentMergedFeatures, pastMergedFeatures,
                                    imgNum, type2s, relatives, type2Archives, currentRedundancyColorer, pastRedundancyColorer, matched=matchedInds)
        else:
            AddNewRedundancy(imgsFeatures, currentImgFeatures, maxWeight, currentMatchedLabelsAssoc, matchedLabels, currentMergedFeatures,
                             pastMergedFeatures, imgNum, type2s, relatives, type2Archives, currentRedundancyColorer, pastRedundancyColorer)
        # if weight <= -0:
        #     print('is New: ' + str((not (True in hasMatched)) or (hasMatched and mergeable)))
        #     print(weight)
        #     testing.ShowChosenRedundancy(currentCCFeatures, pastCCFeatures, winner, COL_WINDOW_SIZE, COL_OVERLAP_SIZE, plotIt=True)
    imgsFeatures = OverwriteImages(imgsFeatures, imgNum, currentMergedFeatures, pastMergedFeatures,
                    matchedLabels, deleteInds, imgsPhraseLabels, imgsNonTextLabels, currentImgFeatures)
    return imgsFeatures


def ProcessPhotos(imgsFeatures, imgsPhraseLabels, imgsNonTextLabels, currentRedundancyColorer, pastRedundancyColorer):
    global redundancyCounter
    # imgsFeatures is a 3d array (not including the numpy arrays inside)
    redundantCC = []
    pastImgFeatures = []
    # type2Archives stores past features that were labeled as type 2 redundancy
    type2Archives = []
    assert len(imgsFeatures) > 0, "len(imgsFeatures) == 0"
    # Copy the first index of imgFeatures to initialize and successively compare with the next indices
    # np array is not deep copied
    pastImgFeatures.append(imgsFeatures[0].copy())
    # For each image...
    for imgNum in range(1, len(imgsFeatures)):
        weights = []
        currentImgFeatures = imgsFeatures[imgNum]
        type2Archives.append([])
        for ind in range(len(currentImgFeatures)):
            # Keep track of the indices when currentImgFeatures was sorted by top
            currentImgFeatures[ind]['index'] = ind
        # Sort by minimum height requirements
        currentImgFeatures = sorted(currentImgFeatures, key=lambda row:row['minHeight'])
        # pastImgInds holds the current index of each pastImgFeatures which was sorted by top
        #    which supplies the need of the each currentImgFeatures at the 'minHeight' key which was also sorted
        currentImgInd, pastImgInds = 0, [0] * imgNum
        # redundantHeap stores all CC in the current image that is considered redundant with
        redundantHeap = []
        # For each connected component in that image...
        while currentImgInd < len(currentImgFeatures):
            currentCCFeatures = currentImgFeatures[currentImgInd]
            left, right, top, bottom = currentCCFeatures['info'][:4]
            assert left<=right and top<=bottom
            minHeightNeed = currentCCFeatures['minHeight']
            maxHeightNeed = bottom + top - minHeightNeed
            # For each previous image...
            for pastImgNum, pastImgInd in enumerate(pastImgInds):
                pastCCFeatures = imgsFeatures[pastImgNum]
                while pastImgInd < len(pastCCFeatures) and pastCCFeatures[pastImgInd]['info'][2] < minHeightNeed:
                    # The needed top is past the pastCCFeatures[pastImgInd] so we increment pastImgInds[pastImgNum] (via
                    #    pastImgInd) since pastImgInds is sorted by the key top which is what is being sought after by
                    #    each element of currentImgFeatures. currentImgFeatures is sorted by the key minHeight
                    #    and is thus sorted by the needs of its elements (supplied by the pastImgInds)
                    pastImgInd += 1
                pastImgInds[pastImgNum] = pastImgInd
                # For each connected component of that previous image...
                while pastImgInd < len(pastCCFeatures) and pastCCFeatures[pastImgInd]['info'][3] <= maxHeightNeed:
                    if FitWithinThresh(pastCCFeatures[pastImgInd]['info'], currentCCFeatures['info']):
                        pastImg = imgsPhraseLabels[pastImgNum] if pastCCFeatures[pastImgInd]['type'] == 'phrase' \
                                                                else imgsNonTextLabels[pastImgNum]
                        currentImg = imgsPhraseLabels[imgNum] if currentCCFeatures['type'] == 'phrase' \
                                                                    else imgsNonTextLabels[imgNum]
                        winner = GetDegreeOfRedundancy(pastCCFeatures[pastImgInd], currentCCFeatures, pastImg, currentImg)
                        if winner['weight'] > 0:
                            weights.append(winner['weight'])
                            # Maxheap based on the given score of redundancy via winner['weight']
                            heapq.heappush(redundantHeap, (-winner['weight'], currentImgInd, pastImgNum, pastImgInd, winner, currentCCFeatures,
                                                           pastCCFeatures[pastImgInd]))
                    # We increment pastImgInd but when the nesting for loop of this while loop runs again, there
                    #    is no change since the succeeding currentImgInd may still be True based on FitWithinThresh
                    #    for this pastCCFeatures[pastImgInds[pastImgNum]]
                    pastImgInd += 1
            currentImgInd += 1
        if weights != []:
            weights = np.array(weights)
            # testing.Summarize(weights)
            absMed = weights - np.median(weights)
            IQR = np.percentile(absMed, 75) - np.percentile(absMed, 25)
            THRESHOLD1 = np.median(weights) * 1.5
            # THRESHOLD1 = np.median(weights) + IQR * 1.5
            print('Max: ', np.max(weights))
            print('Median * 1.5: ', np.median(weights) * 1.5)
            print('Median * 2: ', np.median(weights) * 2)
            print('Median + IQR * 1.5: ', np.median(weights) + IQR * 1.5)
            print('THRESH: ', THRESHOLD1)
            imgsFeatures = UpdateFeatureInfo(imgsFeatures, redundantHeap, imgNum, imgsPhraseLabels, imgsNonTextLabels,
                                           type2Archives[-1], currentRedundancyColorer, pastRedundancyColorer,
                                             currentImgFeatures, THRESHOLD1)
            # testing.ColorRedundancy(currentRedundancyColorer, pastRedundancyColorer, imgsPhraseLabels, imgsNonTextLabels)
    retVal = [{'img': np.where(CC['img'] > 0, 0, 255).astype(np.uint8), 'type': CC['type']} for pic in imgsFeatures for CC in pic]
    testing.WriteColorRedundancy(currentRedundancyColorer, pastRedundancyColorer, imgsPhraseLabels, imgsNonTextLabels)
    # for img in retVal:
    #     print(img['type'])
    #     cv.imshow('img', img['img'].astype(np.uint8))
    #     cv.waitKey()
    #     cv.destroyAllWindows()
    return retVal
        # for i, imgFeatures in enumerate(imgsFeatures[0]):
        #     cv.imshow(str(i), testing.imshow_components(imgFeatures['img'].astype(np.uint8)))
        #     # testing.FullPrint2('print1', imgFeatures['img'])
        #     cv.waitKey()
        #     cv.destroyAllWindows()
        # # print(len(imgsFeatures), len(imgsFeatures[0]))
        # # print(time() - startTime)
        # for i in range(len(redundancyDrawer)):
        #     # pic = np.where(redundancyDrawer[i] == 255, 0, 0).astype(np.uint8)
        #     # pic[np.logical_and(redundancyDrawer[i] != 255, redundancyDrawer[i] != 0)] == 255
        #     cv.imshow('Drawer: ' + str(i), np.where(np.logical_and(redundancyDrawer[i] != 255, redundancyDrawer[i] != 0), 255, 0).astype(np.uint8))
        # #     print(np.any(np.logical_and(redundancyDrawer[i] != 255, redundancyDrawer[i] != 0)))

redundancyCounter = 2

filename = 'learning-single.pkl'
fit = joblib.load(filename)

def main(imgsLabels):
    redundancyDrawer = []
    currentRedundancyColorer = []
    pastRedundancyColorer = []
    maxWidth = 0
    maxHeight = 0
    imgsFeatures = []
    imgsPhraseLabels = []
    imgsNonTextLabels = []
    imgsLabels = [imgLabels for imgLabels in imgsLabels if np.any(imgLabels[0] != 0)]
    numImgs = len(imgsLabels)
    combinations = sorted(list(comb(range(numImgs), 2)))
    redundancyCombs = np.empty((numImgs, numImgs), dtype=np.uint8)
    for i in range(numImgs):
        for j in range(numImgs):
            if i != j:
                redundancyCombs[i, j] = combinations.index(tuple(sorted([i, j]))) + 1
    # imgNum and enumerate only for testing (delete after)
    for imgNum, imgLabels in enumerate(imgsLabels):
        labels, labelsInfo, textLabels, wordLabels, phraseLabels, nonTextLabels = imgLabels
        imgsPhraseLabels.append(phraseLabels)
        imgsNonTextLabels.append(nonTextLabels)
        assert phraseLabels.shape == nonTextLabels.shape
        maxHeight = max(maxHeight, phraseLabels.shape[0])
        maxWidth = max(maxWidth, phraseLabels.shape[1])
        redundancyDrawer.append(np.where(np.logical_or(phraseLabels!=0, nonTextLabels!=0), 255, 0))
        imgFeatures = []
        for ctr in range(2):
            currentLabels = phraseLabels if ctr==0 else nonTextLabels
            labelsInfo = Step5.GetLabelsInfo(currentLabels)
            # if imgNum == 1 and ctr == 1:
            #     cv.imshow('see', np.where(currentLabels == 1, 255, 0).astype(np.uint8))
            #     cv.waitKey()
            for ind, labelInfo in enumerate(labelsInfo):
                imgFeatures.append({})
                imgFeatures[-1]['labelNum'] = ind + 1
                imgFeatures[-1]['type'] = 'phrase' if ctr==0 else 'nonText'
                left, right, top, bottom = labelInfo[:4]
                imgFeatures[-1]['img'] = np.where(currentLabels[top:bottom+1, left:right+1] == ind + 1, ind + 1, 0)
                imgFeatures[-1]['origImg'] = np.zeros((*imgFeatures[-1]['img'].shape, 3), dtype=np.uint16)
                # Record the imgNum, the row in that imgNum (based on the whole image), and the col in that imgNum
                imgFeatures[-1]['origImg'][:, :, 0] = imgNum
                imgFeatures[-1]['origImg'][:, :, 1] = np.arange(top, bottom+1)[:, np.newaxis]
                imgFeatures[-1]['origImg'][:, :, 2] = np.arange(left, right+1)[np.newaxis, :]
                # Include top as the first element of each row.
                # top will be used to identify the possible CCs to compare
                # imgFeatures will also be sorted by top later
                imgFeatures[-1]['top'] = top
                width, area = labelInfo[7], labelInfo[9]
                imgFeatures[-1]['minHeight'] = top - ((1 - MINHEIGHTPROP) * area // width)

                # Assert that currentLabels does not have any other position with the label number ind+1 other
                #    than what is inside the given rectangle
                checkLabels = currentLabels.copy()
                checkLabels[top:bottom+1, left:right+1] = 0
                assert len(checkLabels[checkLabels == ind+1]) == 0

                # Indexing currentLabels is not exactly required by Step6.main() but since inside it, the currentLabels is
                #    again filtered out to only include the ind+1 label and further getting the info of such label (for a general
                #    case instead of passing labelInfo as another parameter), reducing the search space may improve performance
                # Array is also not modified in Step6.main() since it uses np.where at the onset
                # ind + 1 since Step5.GetLabelsInfo() does not include label 0 thus, label 1 has index zero as info
                features = Step6.main(currentLabels[top:bottom+1, left:right+1], ind+1)[0]
                imgFeatures[-1]['info'] = labelInfo
                imgFeatures[-1]['features'] = features
        # Base the sorting on the first element (data about top position) of each row
        imgsFeatures.append(sorted(imgFeatures, key=lambda row:row['top']))
    for ind, img in enumerate(redundancyDrawer):
        redundancyDrawer[ind] = np.pad(img, [(0, maxHeight-img.shape[0]), (0, maxWidth-img.shape[1])],
                                      mode='constant', constant_values=0)
    redundancyDrawer = np.array(redundancyDrawer, dtype=np.uint8)
    currentRedundancyColorer = np.zeros_like(redundancyDrawer, dtype=np.uint16)
    pastRedundancyColorer = np.zeros_like(redundancyDrawer, dtype=np.uint16)
    currentRedundancyColorer = np.array(currentRedundancyColorer)
    pastRedundancyColorer = np.array(pastRedundancyColorer)
    return ProcessPhotos(imgsFeatures, imgsPhraseLabels, imgsNonTextLabels, currentRedundancyColorer, pastRedundancyColorer)
cv.waitKey()
cv.destroyAllWindows()