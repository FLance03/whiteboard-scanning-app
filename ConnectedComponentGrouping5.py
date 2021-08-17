from collections import deque

import numpy as np
import cv2 as cv

from testing import testing

def ContourWithThree(field, copy=True):
    # Returns the field with a label of 3 around the label 1
    if copy:
        field = field.copy()
    rowLen = len(field)
    colLen = len(field[0])
    if rowLen == 1:
        horizontal = field[0]
        for i in range(len(horizontal) - 1):
            if horizontal[i] == 0 and horizontal[i+1] == 1:
                horizontal[i] = 3
    elif colLen == 1:
        for i in range(len(field)):
            if field[i][0] == 0 and field[i+1][0] == 1:
                field[i][0] = 3
    else:
        for row in range(1, rowLen - 1):
            for col in range(1, colLen - 1):
                if field[row, col] == 0 and 1 in field[row - 1:row + 2, col - 1:col + 2]:
                    field[row, col] = 3
        for row in range(1, rowLen - 1):
            if field[row, 0] == 0 and 1 in field[row - 1:row + 2, 0:2]:
                field[row, 0] = 3
            if field[row, colLen - 1] == 0 and 1 in field[row - 1:row + 2, colLen - 2:colLen]:
                field[row, colLen - 1] = 3
        for col in range(1, colLen - 1):
            if field[0, col] == 0 and 1 in field[0:2, col - 1:col + 2]:
                field[0, col] = 3
            if field[rowLen - 1, col] == 0 and 1 in field[rowLen - 2:rowLen, col - 1:col + 2]:
                field[rowLen - 1, col] = 3
        if field[0, 0] == 0 and 1 in [field[0, 1], field[1, 0], field[1, 1]]:
            field[0, 0] = 3
        if field[0, colLen - 1] == 0 and 1 in [field[0, colLen - 2], field[1, colLen - 2], field[1, colLen - 1]]:
            field[0, colLen - 1] = 3
        if field[rowLen - 1, 0] == 0 and 1 in [field[rowLen - 2, 0], field[rowLen - 2, 1], field[rowLen - 1, 1]]:
            field[rowLen - 1, 0] = 3
        if field[rowLen - 1, colLen - 1] == 0 and 1 in [field[rowLen - 2, colLen - 2], field[rowLen - 2, colLen - 1],
                                                        field[rowLen - 1, colLen - 2]]:
            field[rowLen - 1, colLen - 1] = 3
    return field


def FilterPhrasesFromSortedWords(deleteWordsHeadTail, phrasesHeadTail):
    # Returns the filtered head and tail of phrases based on the deleted corresponding words
    # Assumes that the head in deleteWordsHeadTail is sorted
    wordInd, phraseInd = 0, 0
    filterPhrasesHeadTail = []
    while wordInd < len(deleteWordsHeadTail) and phraseInd < len(phrasesHeadTail):
        if deleteWordsHeadTail[wordInd][0] < phrasesHeadTail[phraseInd][0]:
            wordInd += 1
        elif deleteWordsHeadTail[wordInd][0] > phrasesHeadTail[phraseInd][0]:
            filterPhrasesHeadTail.append(phrasesHeadTail[phraseInd])
            phraseInd += 1
        else:
            wordInd += 1
            phraseInd += 1
    filterPhrasesHeadTail = np.array(filterPhrasesHeadTail)
    return filterPhrasesHeadTail


def GetLabelsInfo(labels, numLabels=None):
    # numLabels contains the number of labels including the background label 0
    lastIndLabel = np.max(labels) if numLabels==None else numLabels-1
    # Holds the information for each connected component: 4 points of rectangle (left right top bottom),
    # 2 for each of the x-coord for both the top and bottom seeds for the label,
    # 1 for num of black pixels in rect, 1 for width, 1 for height, 1 for num of pixels in rect
    labelsInfo = np.zeros((lastIndLabel, 10), np.uint32)
    # To check whether an info for a particular label has been updated (since they all start zero which supposedly
    # precedes over all possible starting coordinates of a rectangle
    labelTrigger = np.zeros(lastIndLabel, np.uint8)
    for rowInd, rowVal in enumerate(labels):
        for colInd, labelNum in enumerate(rowVal):
            if labelNum != 0:
                if not labelTrigger[labelNum - 1]:
                    labelTrigger[labelNum - 1] = 1
                    # index: 0 is left of rect, 2 is top of rect, 4 is the x-coord of top seed
                    # Numpy [[labelNum-1],[0,2,4]] => [[labelNum-1,labelNum-1,labelNum-1],[0,2,4]] => [labelNum-1,0],[labelNum-1,2],[labelNum-1,4]
                    labelsInfo[[labelNum - 1], [0, 2, 4]] = colInd, rowInd, colInd
                elif colInd < labelsInfo[labelNum - 1][0]:
                    # Only the left of the rectangle may be updated since the loop strictly goes top to bottom unlike on the horizontal
                    # side which may go back to the left. Thus, the top of the rectangle doesnt change and the there cannot
                    # be a "more top" (greater/higher y-coord) top seed, solidifying the x-coord info kept
                    labelsInfo[labelNum - 1][0] = colInd
                if colInd > labelsInfo[labelNum - 1][1]:
                    # Update right of rect
                    labelsInfo[labelNum - 1][1] = colInd
                if rowInd > labelsInfo[labelNum - 1][3]:
                    # Update bottom of rect and x-coord of the bottom seed
                    labelsInfo[labelNum - 1][3] = rowInd
                    labelsInfo[labelNum - 1][5] = colInd
                # Increment counter for black pixels for a label
                labelsInfo[labelNum - 1][6] += 1
    # Width = right - left + 1
    labelsInfo[:, 7] = labelsInfo[:, 1] - labelsInfo[:, 0] + 1
    # Height = bottom - top + 1
    labelsInfo[:, 8] = labelsInfo[:, 3] - labelsInfo[:, 2] + 1
    # Num of Pixels (Area of Rect) = width * height
    labelsInfo[:, 9] = labelsInfo[:, 7] * labelsInfo[:, 8]
    return labelsInfo


def PreFilter(labels, numLabels=None, copy=True):
    if copy:
        labels = labels.copy()
    if numLabels == None:
        numLabels = np.max(labels) + 1
    # Remove labels with counts less than some threshold
    # countLabels holds the number of occurences of a certain label: f(label num - 1) => how many label num are there
    countLabels = np.histogram(labels, np.arange(1, numLabels + 1))[0]
    excludeLabelNums = countLabels < 6
    # backTrackNums holds the new label being indexed by the old one: f(old label num - 1) => new label num
    backTrackNums = np.empty_like(excludeLabelNums, np.uint16)
    newLabel = 1
    for ind, isExclude in enumerate(excludeLabelNums):
        if isExclude:
            backTrackNums[ind] = 0
        else:
            backTrackNums[ind] = newLabel
            newLabel += 1
    # Update the label nums in labels var
    for rowInd, rowVal in enumerate(labels):
        for colInd, labelNum in enumerate(rowVal):
            if labelNum != 0:
                labels[rowInd][colInd] = backTrackNums[labelNum - 1]
    return labels


def Filter(labelsInfo, attrInd):
    attrs = labelsInfo[:, attrInd]
    minAttr = np.min(attrs)
    maxAttr = np.max(attrs)
    if maxAttr-minAttr < 500:
        bins = np.linspace(minAttr, maxAttr + 1, len(attrs)//3)
    else:
        # np.max()+step to include the max in array
        step = -16 * np.log(len(attrs)) + 100
        if step < 3:
            step = 3
        bins = np.arange(minAttr, maxAttr + step, step)
    histo = np.histogram(attrs, bins)[0]
    # Merge some bins such that the value of the bin equals the sum of it and the ones beside it
    blurredHisto = histo[:-2] + histo[1:-1] + histo[2:]
    if len(blurredHisto) == 0:
        return labelsInfo
    else:
        blurredHistoMaxInd = np.argmax(blurredHisto)
        # If blurredHistoMaxInd held 0, this would come from the sum of index 1 in histo and those beside it
        mostPopulatedLen = histo[blurredHistoMaxInd] + histo[blurredHistoMaxInd + 1] + histo[blurredHistoMaxInd + 2]
        # Average of most populated by accessing back those which belongs to the representative bins
        mostPopulatedAttrVal = 0
        low, high = bins[blurredHistoMaxInd], bins[blurredHistoMaxInd + 3]
        for attr in attrs:
            # Get all attrs within the bins belonging to the "most populated attr", inclusive of the bins
            # histo[0] has the number of occurences between bins[0] (inclusive) to bins[1] (exclusive)
            # If most populated are indices 0, 1, 2 of histo (0 to 1, 1 to 2, and 2 to 3 in bins), the low is bins[0] and high is bins[3]
            # Since the last index of histo has both bins inclusive,, just generalized to all being inclusive (<=)
            if low <= attr <= high:
                mostPopulatedAttrVal += attr
        aveMostPopulated = mostPopulatedAttrVal / mostPopulatedLen
        aveAttrVal = sum(attrs) / len(attrs)
        graterAve = max(aveMostPopulated, aveAttrVal)
        return labelsInfo[labelsInfo[:, attrInd] < graterAve * 5]


def GetInvolvedPoints(shape, points, line, R, clusterSize=1):
    diagSize = (shape[0]**2 * shape[1]**2)**(1/2)
    rho, theta = line
    # R is the bin size
    binLen = np.ceil(diagSize/R)
    # Get the bin index from the given rho
    rhoInd = rho // R
    # Odd clusters have the topInd and bottomInd intervals equidistant with the rho Ind
    # Even clusters have a topInd farther than that of the bottomInd from the rho Ind
    topInd, bottomInd = rhoInd + clusterSize//2, rhoInd - (clusterSize - clusterSize//2 - 1)
    # Thresholds used in comparison so would not matter if beyond minimum or maximum
    topThresh, bottomThresh = (topInd + 1) * R, bottomInd * R

    p = points[:, 1] * np.sin(theta) + points[:, 0] * np.cos(theta)
    clusterInd = np.bitwise_and(bottomThresh <= p, p <= topThresh)

    return clusterInd


def DistanceBetweenLabels(labels, leftBox, rightBox):
    # Returns the least number of eight-connected pixel distance between the two groups
    left, right, top, bottom = leftBox[0], rightBox[1], min(leftBox[2], rightBox[2]), max(leftBox[3], rightBox[3])
    field = labels[top:bottom+1, left:right+1]
    # Get the label at each box using the top of rect (y-coor) and the top seed (x-coor)
    leftLabel = labels[leftBox[2], leftBox[4]]
    rightLabel = labels[rightBox[2], rightBox[4]]
    for row, fieldRow in enumerate(field):
        for col, label in enumerate(fieldRow):
            if label == leftLabel:
                field[row, col] = 1
            elif label == rightLabel:
                field[row, col] = 2
            elif label != 0:
                field[row, col] = 0
    ContourWithThree(field, copy=False)
    queue = deque()
    for row in range(len(field)):
        for col in range(len(field[0])):
            if field[row][col] == 3:
                queue.appendleft((row, col))
    while len(queue) > 0:
        row, col = queue.pop()
        dist = field[row][col] + 1
        l, r, t, b = col-1, col+1, row-1, row+1
        if l <= -1:
            l = 0
        if r >= len(field[0]):
            r = len(field[0]) - 1
        if t <= -1:
            t = 0
        if b >= len(field):
            b = len(field) - 1
        for i in range(t, b+1):
            for j in range(l, r+1):
                if field[i, j] == 2:
                    # dist - 3 since started at label 3 which represents distance 0
                    return dist - 3
                if field[i, j] == 0:
                    field[i, j] = dist
                    queue.appendleft((i, j))
    return -1


def FilterHeadTail(wordsHeadTail, phrasesHeadTail):
    deleteWordsHeadTail = np.tile(False, len(wordsHeadTail))
    for ind in range(len(wordsHeadTail) - 2):
        if np.all(wordsHeadTail[ind: ind + 3, 1] == 2):
            # Remove 3 consecutive isolated characters
            # deleteWordsHeadTail records the indices of such characters
            deleteWordsHeadTail[ind: ind + 3] = True
    filterWordsHeadTail = wordsHeadTail[np.logical_not(deleteWordsHeadTail)]
    # Change deleteWordsHeadTail from having the indices to accessing the actual elements to be deleted
    deleteWordsHeadTail = wordsHeadTail[deleteWordsHeadTail]
    filterPhrasesHeadTail = FilterPhrasesFromSortedWords(deleteWordsHeadTail, phrasesHeadTail)
    deleteWordsHeadTail = np.tile(False, len(filterWordsHeadTail))
    for ind, word in enumerate(filterWordsHeadTail):
        if word[1] == 1 and word[2] - word[1] < 2:
            # Remove isolated words with less than 3 characters
            deleteWordsHeadTail[ind] = True
    filterWordsHeadTail = filterWordsHeadTail[np.logical_not(deleteWordsHeadTail)]
    deleteWordsHeadTail = filterWordsHeadTail[deleteWordsHeadTail]
    filterPhrasesHeadTail = FilterPhrasesFromSortedWords(deleteWordsHeadTail, filterPhrasesHeadTail)
    return filterWordsHeadTail, filterPhrasesHeadTail


img = cv.imread('./testing/iotbinarized.jpg', 0)
# img = img[90:350, 40:350]
img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)[1]
notImg = cv.bitwise_not(img)

numLabels, labels = cv.connectedComponents(notImg, connectivity=8)
print(labels.dtype)
PreFilter(labels, copy=False)
coloredLabels = testing.imshow_components(labels)

labelsInfo = GetLabelsInfo(labels)
# Filter based on area (9th index)
filterInfo = Filter(labelsInfo, 9)
widthsInfo, heightsInfo = filterInfo[:, 7], filterInfo[:, 8]
# Exclude (filter) those that are 10 times wider than their height or 20 times taller than their width
filterInfo = filterInfo[np.logical_and(widthsInfo*20 > heightsInfo, heightsInfo*10 > widthsInfo)]

# Filter again based on height by first appending the height data
# filterInfo = Filter(filterInfo, 8)
# filterInfo = filterInfo[filterInfo[:, 8] > 3]

# Center = ( (left+right)/2 , (top+bottom)/2 ) but doubled to avoid decimal numbers
centers = np.vstack(((filterInfo[:, 1] + filterInfo[:, 0])//2, (filterInfo[:, 3] + filterInfo[:, 2])//2)).transpose()

pointsPlain = np.zeros((notImg.shape[0], notImg.shape[1]), np.uint8)
for center in centers:
    pointsPlain[center[1]][center[0]] = 255
aveHeight = np.mean(filterInfo[:, 8])
R = 0.2 * aveHeight
trig=0
outPoints = centers
for count in range(2):
    for runningThreshold in range(20, 2, -1):
        lines = cv.HoughLines(pointsPlain, R, np.pi / 180, runningThreshold, None, 0, 0)
        if lines is not None:
            lines = lines.reshape(-1, 2)
            if count == 0:
                # Check first all lines with theta of 90 +- 5 degrees or 2.5 +- 2.5 degrees or 177.5 +- 2.5 degrees
                # Check first all lines with theta of pi/2 +- 5*pi/180 or 2.5*pi/180 +- 2.5*pi/180 or 177.5*pi/180 +- 2.5*pi/180
                horizontal = np.bitwise_and(17*np.pi/36 <= lines[:, 1], lines[:, 1] <= 19*np.pi/36)
                vertical = np.bitwise_and(0 <= lines[:, 1], lines[:, 1] <= np.pi/36)
                vertical = np.bitwise_or(vertical, np.bitwise_and(35*np.pi/36 <= lines[:, 1], lines[:, 1] <= np.pi))
                lines = lines[np.bitwise_or(horizontal, vertical)]
            for line in lines:
                rho, theta = line
                inPointInds = GetInvolvedPoints(pointsPlain.shape, outPoints, (rho, theta), R, clusterSize=11)
                # Clustering factor = Average height in cluster / R
                Ha = np.mean(filterInfo[inPointInds, 8])
                newClusFactor = np.round(Ha / R)
                # For clustering +-newClusFactor cells, around the primary cell
                newClusFactor = 2*newClusFactor + 1
                inPointInds2 = GetInvolvedPoints(pointsPlain.shape, outPoints, (rho, theta), R, clusterSize=newClusFactor)
                inPointsInfo = filterInfo[inPointInds2]
                inPointsCenters = centers[inPointInds2]
                # Sort included points by the left of the rectangle which is at the 0th index of a row
                sortInd = np.argsort(inPointsInfo[:, 0])
                inPointsInfo = inPointsInfo[sortInd]
                inPointsCenters = inPointsCenters[sortInd]
                # inPointsCenters = np.r_[inPointsCenters[:22], np.array([[584, 213]]), inPointsCenters[22:], np.array([[1500, 1500]])]
                print(inPointsCenters)
                # distances = []
                # for i in range(len(inPointsInfo) - 1):
                #     distances.append(DistanceBetweenLabels(labels, inPointsInfo[i, 0:5], inPointsInfo[i+1, 0:5]))
                distances = []
                for i in range(len(inPointsCenters) - 1):
                    x = (inPointsCenters[i+1, 0].astype(np.int64) - inPointsCenters[i, 0].astype(np.int64))**2
                    y = (inPointsCenters[i+1, 1].astype(np.int64) - inPointsCenters[i, 1].astype(np.int64))**2
                    distances.append((x+y)**(1/2))
                distances = np.array(distances)
                print(distances)
                print(2.5 * Ha)
                # Use the height dimension if line is within -45 to 45 degrees (inclusive) else use width
                dimensionInd = 8 if -3*np.pi/4 <= theta <= 3*np.pi/4 else 7
                dimensionVals = inPointsInfo[:, dimensionInd]
                # For each element, get the average with the four nearest neighbors
                if len(dimensionVals) < 5:
                    # Get the mean of all elements at that dimension and tile the array to have len(dimensionVals) elements
                    dimensionMeans = np.tile(np.mean(dimensionVals), len(dimensionVals))
                else:
                    runningSum = sum(dimensionVals[0:5])
                    dimensionMeans = np.empty_like(dimensionVals, dtype=np.float32)
                    dimensionMeans[:3] = runningSum / 5
                    for i in range(3, len(dimensionMeans)-2):
                        runningSum = runningSum - dimensionVals[i-3] + dimensionVals[i+2]
                        dimensionMeans[i] = runningSum / 5
                    dimensionMeans[-3:] = runningSum / 5
                wordCount, phraseCount = 0, 0
                words, phrases = np.empty_like(dimensionMeans, dtype=np.uint32), np.empty_like(dimensionMeans, dtype=np.uint32)
                words[0] = phrases[0] = 0
                print(len(distances), len(dimensionMeans))
                # First and third index of inner list are the head and tail respectively
                # Second index of the inner list represents the type: 0 => isolated, 1 => word, 2 => phrase
                # Word if not part of a phrase but also not isolated.
                # Phrase if part of phrase even if includes single character words.
                wordsHeadTail = [[0, 0]]
                phrasesHeadTail = [[0]]
                for ind, distance in enumerate(distances):
                    if distance >= 2.5 * Ha:
                        wordsHeadTail[-1].append(ind)
                        phrasesHeadTail[-1].append(ind)
                        wordsHeadTail.append([ind + 1, 0])
                        phrasesHeadTail.append([ind + 1])
                        wordCount += 1
                        phraseCount += 1
                    elif distance > dimensionMeans[ind + 1]:
                        # Word ends so record the tail
                        wordsHeadTail[-1].append(ind)
                        # Since distance < 2.5 * Ha but > mean, part of phrase not word thus, assign as 2
                        wordsHeadTail[-1][1] = 2
                        # Since distance < 2.5 * Ha but > mean, the next one is still part of the phrase
                        wordsHeadTail.append([ind + 1, 2])
                        wordCount += 1
                    elif wordsHeadTail[-1][1] != 2:
                        # Close enough to be considered a word and the group still not considered part of a phrase
                        wordsHeadTail[-1][1] = 1
                    words[ind + 1] = wordCount
                    phrases[ind + 1] = phraseCount
                wordsHeadTail[-1].append(ind + 1)
                phrasesHeadTail[-1].append(ind + 1)
                wordsHeadTail = np.array(wordsHeadTail)
                filterWordsHeadTail, filterPhrasesHeadTail = FilterHeadTail(wordsHeadTail, phrasesHeadTail)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(coloredLabels, pt1, pt2, (255,255,255), 1, cv.LINE_AA)
                trig+=1
                break
            trig==1
            break
    if trig==1:
        break

# print(len(centers[inPointInds]))
# print(len(centers[inPointInds2]))
# for x, y in centers[inPointInds]:
#     cv.circle(coloredLabels, (x, y), 5, (0, 255, 0), -1)
# for x, y in centers[inPointInds2]:
#     cv.circle(coloredLabels, (x, y), 3, (0, 0, 255), -1)

# for left, right, top, bottom, *other in filterInfo[inPointInds2]:
#     cv.rectangle(coloredLabels, (left, top), (right, bottom), (255, 255, 255), 1)


# cv.imshow('Left', img)
# cv.imshow('LeftNot', coloredLabels)
# testing.PlotIt(coloredLabels)

cv.waitKey()
cv.destroyAllWindows()

