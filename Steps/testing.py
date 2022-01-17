import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import Steps.SaturationAndOutlier2 as Step2


def PlotIt(img, title=None):
    plt.subplot(1, 1, 1)
    plt.imshow(img, 'gray') if len(img.shape) == 2 else plt.imshow(img)
    plt.title('Image')
    if title is not None:
        plt.suptitle(title)
    plt.show()

def GetLabelsInfo(labels, numLabels=None):
    # numLabels contains the number of labels including the background label 0
    lastIndLabel = np.max(labels) if numLabels is None else numLabels - 1
    retVal = np.zeros((lastIndLabel, 10), dtype=np.uint32)
    for labelNum in range(1, lastIndLabel + 1):
        ver, hor = np.where(labels == labelNum)
        if ver.size > 0 and hor.size > 0:
            minVer, minHor = np.min(ver), np.min(hor)
            maxVer, maxHor = np.max(ver), np.max(hor)
            topSeed = np.min(np.where(labels[minVer, :] == labelNum))
            bottomSeed = np.min(np.where(labels[maxVer, :] == labelNum))
            blackCount = ver.size
            height = maxVer - minVer + 1
            width = maxHor - minHor + 1
            area = width * height
            retVal[labelNum - 1] = [minHor, maxHor, minVer, maxVer, topSeed, bottomSeed, blackCount, width, height, area]
    return retVal

def cropOutEdges(img):
    assert img is not None, "Image does not exist"
    img = Step2.main(img)
    assert img is not None, "Image does not exist"
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binarized = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    binarized = np.where(binarized == 255, 0, 1)
    left, right, top, bottom = GetLabelsInfo(binarized)[0, :4]
    binarized = np.where(binarized == 0, 255, 0)
    return binarized[top:bottom+1, left:right+1].astype(np.uint8)

def main(img):
    # PlotIt(cropOutEdges(cv.imread(read)))
    return cropOutEdges(img)

# cv.imwrite('..1.png', main(img=cv.imread('testing.png')))