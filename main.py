from time import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from Steps import (Preprocessing1 as Step1,
                   SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5,
                   FeatureExtraction6 as Step6,
                   Redundancy7 as Step7,
                   ConvertToDocx as Step8)
from testing import testing

np.seterr(all='raise')
startTime = time()

testImages = []
anded = []
imgsLabels = []
i = 0
while True:
    img = cv.imread('./Server/' + str(i) + '.jpg')
    if img is None:
        break
    # cv.imshow(str(i), testing.ResizeWithAspectRatio(img, height=500))
    # cv.waitKey()
    # cv.destroyAllWindows()
    # print(img)
    testImages.append(img)
    i += 1
for ind, testImage in enumerate(testImages):
    img = testImages[ind]
    assert img is not None
    # cv.imshow('Original', testing.ResizeWithAspectRatio(img, height=500))
    # cv.waitKey()
    # cv.destroyAllWindows()
    # if ind not in [1]:
    img = Step1.Preprocessing1(img)
    cv.imshow('Original', testing.ResizeWithAspectRatio(img, height=500))
    # cv.waitKey()
    # cv.destroyAllWindows()
    img = Step2.main(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, dsize=(int(gray.shape[1] * 500 / float(gray.shape[0])), 500), interpolation=cv.INTER_AREA)
    _, bw = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    kernel = np.ones((5, 5),np.uint8)
    cv.imshow('Before', testing.ResizeWithAspectRatio(img, height=500))
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    cv.imshow('After', testing.ResizeWithAspectRatio(img, height=500))
    cv.waitKey()
    cv.destroyAllWindows()
    cv.imwrite(str(ind) + 'p.jpg', bw)
    cv.imwrite(str(ind) + 'n.jpg', bw)
    cv.imwrite(str(ind) + 'o.jpg', bw)
    bw = bw[15:-15]
    # anded.append(np.where(bw == 0, 1, 0))
    # cv.imshow('Original', testing.ResizeWithAspectRatio(img, height=500))
    # cv.waitKey()
    # cv.destroyAllWindows()
    # Remove the possible lines from the blackboard edges
    # labels, labelsInfo, textLabels, wordLabels, phraseLabels, nonTextLabels
    imgsLabels.append(Step5.main(bw))
listCC = Step7.main(imgsLabels)
Step8.ConvertToDocx(listCC)
print('Time: ', time() - startTime)


