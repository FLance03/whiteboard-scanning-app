from pprint import pprint as p

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# dir = 'testing/pics and texts/'
dir = './'

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

def WriteFile(someObject, pretty=False):
    with open(dir + 'print2.txt', 'w') as out:
        print(someObject, file=out) if not pretty else p(someObject, stream=out)

def FullPrint(*args, **kwargs):
  opt = np.get_printoptions()
  np.set_printoptions(threshold=np.inf)
  with open(dir + 'print.txt', 'w') as out:
    print(*args, **kwargs, file=out)
  np.set_printoptions(**opt)

def FullPrint2(filename, *args, **kwargs):
  opt = np.get_printoptions()
  np.set_printoptions(threshold=np.inf)
  with open(dir + filename + '.txt', 'w') as out:
    print(*args, **kwargs, file=out)
  np.set_printoptions(**opt)

def imshow_components(labels):
    if np.max(labels) == 0:
        return labels.astype(np.uint8)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img
def PlotIt(img, title=None):
    plt.subplot(1, 1, 1)
    plt.imshow(img, 'gray') if len(img.shape) == 2 else plt.imshow(img)
    plt.title('Image')
    if title is not None:
        plt.suptitle(title)
    plt.show()

# def ShowChosenRedundancy(currentCCFeatures, pastCCFeatures, comparison):
#     t = currentCCFeatures['info'][2] if currentCCFeatures['info'][2] > pastCCFeatures['info'][2] else \
#     pastCCFeatures['info'][2]
#     b = currentCCFeatures['info'][3] if currentCCFeatures['info'][3] < pastCCFeatures['info'][3] else \
#     pastCCFeatures['info'][3]
#     l = currentCCFeatures['info'][0] if currentCCFeatures['info'][0] > pastCCFeatures['info'][0] else \
#     pastCCFeatures['info'][0]
#     r = currentCCFeatures['info'][1] if currentCCFeatures['info'][1] > pastCCFeatures['info'][2] else \
#     pastCCFeatures['info'][1]
#     cv.imshow('current', imshow_components(
#         currentCCFeatures['img'][t - currentCCFeatures['info'][2]:b - currentCCFeatures['info'][2] + 1,
#         comparison['currentLeft']:comparison['currentRight'] + 1]))
#     cv.imshow('past', imshow_components(
#         pastCCFeatures['img'][t - pastCCFeatures['info'][2]:b - pastCCFeatures['info'][2] + 1,
#         comparison['pastLeft']:comparison['pastRight'] + 1]))
#     PlotIt(imshow_components(pastCCFeatures['img']))
#     cv.waitKey()
#     cv.destroyAllWindows()

def ShowChosenRedundancy(currentCCFeatures, pastCCFeatures, comparison, COL_WINDOW_SIZE, COL_OVERLAP_SIZE, plotIt=False):
    nonOverlapSize = COL_WINDOW_SIZE - COL_OVERLAP_SIZE
    currentLeft = (comparison['currentLeft'] + comparison['redundancyOffset'][0]) * nonOverlapSize + COL_WINDOW_SIZE
    currentRight = (comparison['currentLeft'] + comparison['redundancyOffset'][1]) * nonOverlapSize + COL_WINDOW_SIZE
    cv.imshow('current', imshow_components(
        currentCCFeatures['img'][comparison['currentTop']:
                                 comparison['currentTop'] + currentCCFeatures['info'][8],
                                    currentLeft:
                                    currentRight]))
    pastLeft = (comparison['pastLeft'] + comparison['redundancyOffset'][0]) * nonOverlapSize + COL_WINDOW_SIZE
    pastRight = (comparison['pastLeft'] + comparison['redundancyOffset'][1]) * nonOverlapSize + COL_WINDOW_SIZE
    cv.imshow('past', imshow_components(
        pastCCFeatures['img'][comparison['pastTop']:
                              comparison['pastTop'] + pastCCFeatures['info'][8],
                                pastLeft:
                                pastRight]))
    if plotIt:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Left, Right, Top, Bottom of "Winner"')
        ax1.set_title('Past')
        ax1.imshow(imshow_components(pastCCFeatures['img']))
        ax2.set_title('Current')
        ax2.imshow(imshow_components(currentCCFeatures['img']))
        plt.show()
    cv.waitKey()
    cv.destroyAllWindows()


def ShowCurrentPast(currentCCFeatures, pastCCFeatures):
    cv.imshow('current image', imshow_components(currentCCFeatures['img']))
    cv.imshow('past image', imshow_components(pastCCFeatures['img']))
    cv.waitKey()
    cv.destroyAllWindows()

def ColorRedundancy(currentRedundancyColorer, pastRedundancyColorer, imgsPhraseLabels, imgsNonTextLabels):
    # redundancyColorer = currentRedundancyColorer.copy().astype(np.uint8)
    redundancyColorer = np.concatenate((currentRedundancyColorer, pastRedundancyColorer), axis=0).astype(np.uint8)
    uniqueLabels = np.unique(redundancyColorer)
    maxLabel = np.max(uniqueLabels)
    uniqueLabelsLen = len(uniqueLabels)
    wholePics = []
    if maxLabel > 1:
        # Remap labels starting at label 2 by the element half array size away from it Ex: [0,1,2,3,4,5] -> [0,1,4,2,5,3]
        mapLabels = np.insert(uniqueLabels[:(uniqueLabelsLen - 1)//2 + 1],
                              range(1, uniqueLabelsLen - (uniqueLabelsLen - 1)//2),
                              uniqueLabels[(uniqueLabelsLen - 1)//2 + 1:])
        occupiedLabels = mapLabels[np.searchsorted(uniqueLabels, redundancyColorer)]
        occupiedLabels = occupiedLabels * (179 / maxLabel)
        for i in range(len(redundancyColorer)):
            label_hue = np.uint8(179*occupiedLabels[i]/maxLabel)
            blank_ch = 255*np.ones_like(label_hue)
            labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
            labeled_img[label_hue==0] = 0
            colored = labeled_img
            # currImg holds all images in BGR format where those with writings are white
            currImg = np.zeros((*imgsNonTextLabels[i % len(currentRedundancyColorer)].shape, 3), dtype=np.uint8)
            currImg[imgsNonTextLabels[i % len(currentRedundancyColorer)].nonzero()] = (255, 255, 255)
            currImg[imgsPhraseLabels[i % len(currentRedundancyColorer)].nonzero()] = (255, 255, 255)
            # currImg = np.zeros((*imgsNonTextLabels[i].shape, 3), dtype=np.uint8)
            # currImg[imgsNonTextLabels[i].nonzero()] = (255, 255, 255)
            # currImg[imgsPhraseLabels[i].nonzero()] = (255, 255, 255)
            wholePic = np.where(np.logical_and(redundancyColorer[i, :, :, np.newaxis]!=0, currImg==255),
                                colored, currImg)
            wholePics.append(wholePic)
            # cv.imshow('Colorer: ' + str(i), wholePic)
        cv.waitKey()
        cv.destroyAllWindows()
    return wholePics

def WriteColorRedundancy(currentRedundancyColorer, pastRedundancyColorer, imgsPhraseLabels, imgsNonTextLabels):
    # wholePics = ColorRedundancy(currentRedundancyColorer, pastRedundancyColorer, imgsPhraseLabels, imgsNonTextLabels)
    wholePics = ColorRedundancy(currentRedundancyColorer, pastRedundancyColorer, imgsPhraseLabels, imgsNonTextLabels)
    np.savez('output.npz', labels=np.concatenate((currentRedundancyColorer, pastRedundancyColorer), axis=0).astype(np.uint8)
                            , colors=np.array(wholePics))
    # for imgNum, wholePic in enumerate(wholePics):
    #     cv.imwrite(str(imgNum) + '.png', wholePic)

def Summarize(data):
    df = pd.DataFrame(data, columns=['Values'])
    print(df.describe())
    max = np.max(data)
    min = np.min(data)
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    print('Range', max - min)
    print('IQR: ', IQR)
    print('Left Outlier: ', np.percentile(data, 25) - 1.5 * IQR)
    print('Right Outlier: ', np.percentile(data, 75) + 1.5 * IQR)
    print(data)
    print(np.median(data))
    print(data - np.median(data))
    print(np.abs(data - np.median(data)))
    print('Median Abs Dev: ')
    df = pd.DataFrame(np.abs(data - np.median(data)), columns=['Deviation'])
    print(df.describe())