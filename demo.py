import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from Steps import (SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5,
                   FeatureExtraction6 as Step6)
from testing import testing

img = cv.imread('./testing/pics and texts/' + 'iot' + '.jpg')
img = Step2.main(img, CELLGROUPSIZE=30, CELLSIZE=15)
cv.imshow('Step 2', img)
cv.waitKey()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
cv.imshow('Step 3: Binarization', bw)
cv.waitKey()
# # Remove the possible lines from the blackboard edges
labels, labelsInfo, textNonText, textLabels, wordLabels, phraseLabels, nonTextLabels = Step5.main(bw)
cv.imshow('Step 5: Texts', testing.ResizeWithAspectRatio(testing.imshow_components(wordLabels), width=455))
cv.imshow('Step 5: Non-Texts', testing.ResizeWithAspectRatio(testing.imshow_components(nonTextLabels), width=455))
cv.waitKey()

cv.destroyAllWindows()


#     np.savez(testImage + '.npz', labels=labels, labelsInfo=labelsInfo, textNonText=textNonText, textLabels=textLabels,
#              wordLabels=wordLabels, phraseLabels=phraseLabels, nonTextLabels=nonTextLabels)


