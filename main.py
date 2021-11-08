from time import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from Steps import (SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5,
                   FeatureExtraction6 as Step6)
from testing import testing

np.seterr(all='raise')
startTime = time()

testImages = ['1s', '2s', '3s', '4s', '5s', '6s']
anded = []
for ind, testImage in enumerate(testImages):
    img = cv.imread('./testing/pics and texts/Group Tests Step1/2/' + testImage + '.jpg')
    assert img is not None
    img = Step2.main(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, dsize=(500, 500), interpolation=cv.INTER_LINEAR)
    _, bw = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    bw = bw[15:-15]
    # anded.append(np.where(bw == 0, 1, 0))
    # cv.imshow(str(ind + 1), bw)
    # Remove the possible lines from the blackboard edges
    labels, labelsInfo, textLabels, wordLabels, phraseLabels, nonTextLabels = Step5.main(bw)

    np.savez(testImage + '.npz', labels=labels, labelsInfo=labelsInfo, textLabels=textLabels,
             wordLabels=wordLabels, phraseLabels=phraseLabels, nonTextLabels=nonTextLabels)
print(time() - startTime)
# cv.imshow('first second', np.where(np.bitwise_and(anded[0], anded[1]) == 1, 0, 255).astype(np.uint8))
# cv.imshow('first third', np.where(np.bitwise_and(anded[0], anded[2]) == 1, 0, 255).astype(np.uint8))
# cv.imshow('second third', np.where(np.bitwise_and(anded[1], anded[2]) == 1, 0, 255).astype(np.uint8))
cv.waitKey()
cv.destroyAllWindows()
# imgsFeatures = []
# # Start loop from the most recent picture
# for img in testImages[::-1]:
#     file = np.load(img + '.npz')
#     labels, labelsInfo, textNonText, textLabels, wordLabels, phraseLabels, nonTextLabels = \
#         file['labels'], file['labelsInfo'], file['textNonText'], file['textLabels'], file['wordLabels'], file[
#             'phraseLabels'], file['nonTextLabels']
#     phraseLabelsInfo = Step5.GetLabelsInfo(phraseLabels)
#     testing.FullPrint2('print3', phraseLabels)
#     imgFeatures = []
#     for ind, phraseLabelInfo in enumerate(phraseLabelsInfo):
#         imgFeatures.append([])
#         left, right, top, bottom = phraseLabelInfo[:4]
#         # Include top as the first element of each row.
#         # top will be used to identify the possible CCs to compare
#         # imgFeatures will also be sorted by top later
#         imgFeatures[-1].append(top)
#
#         # Assert that phraseLabels does not have any other position with the label number ind+1 other
#         #    than what is inside the given rectangle
#         checkLabels = phraseLabels.copy()
#         checkLabels[top:bottom+1, left:right+1] = 0
#         assert len(checkLabels[checkLabels == ind+1]) == 0
#
#         # Indexing phraseLabels is not exactly required by Step6.main() but since inside it, the phraseLabels is
#         #    again filtered out to only include the ind+1 label and further getting the info of such label (for a general
#         #    case instead of passing labelInfo as another parameter), reducing the search space may improve performance
#         # Array is also not modified in Step6.main() since it uses np.where at the onset
#         # ind + 1 since Step5.GetLabelsInfo() does not include label 0 thus, label 1 has index zero as info
#         features, info = Step6.main(phraseLabels[top:bottom+1, left:right+1], ind+1)
#         imgFeatures[-1].append(info)
#         imgFeatures[-1].append(features)
# # Base the sorting on the first element (data about top position) of each row
# imgsFeatures.append(sorted(imgFeatures, key=lambda row:row[0]))
#
# # phraseLabelsInfo = Step5.GetLabelsInfo(phraseLabels)
# # features = Step6.main(phraseLabels, phraseLabelsInfo[0], 1)
# # testing.FullPrint(features)

cv.imshow('Labels', testing.imshow_components(labels))
# cv.imshow('Labels2', testing.imshow_components(np.where(phraseLabels == 1, phraseLabels, 0)))
cv.imshow('Texts', testing.imshow_components(textLabels))
# cv.imshow('Words', testing.imshow_components(wordLabels))
cv.imshow('Phrases', testing.imshow_components(phraseLabels))
cv.imshow('Non Texts', testing.imshow_components(nonTextLabels))
#
cv.waitKey()
cv.destroyAllWindows()

