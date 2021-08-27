import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from Steps import (SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5,
                   FeatureExtraction6 as Step6)
from testing import testing

# img = cv.imread('./testing/pics and texts/redundancyfrontcropped.jpg')
# img = Step2.main(img)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# _, bw = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
# # Remove the possible lines from the blackboard edges
# labels, labelsInfo, textNonText, textLabels, wordLabels, phraseLabels, nonTextLabels = Step5.main(bw)
#
# np.savez('Step5Outputs2.npz', labels=labels, labelsInfo=labelsInfo, textNonText=textNonText, textLabels=textLabels,
#          wordLabels=wordLabels, phraseLabels=phraseLabels, nonTextLabels=nonTextLabels)
file = np.load('Step5Outputs2.npz')
labels, labelsInfo, textNonText, textLabels, wordLabels, phraseLabels, nonTextLabels = \
    file['labels'], file['labelsInfo'], file['textNonText'], file['textLabels'], file['wordLabels'], file[
        'phraseLabels'], file['nonTextLabels']
phraseLabelsInfo = Step5.GetLabelsInfo(phraseLabels)
features = Step6.main(phraseLabels, phraseLabelsInfo[0], 1)
testing.FullPrint(features)

# cv.imshow('Labels', testing.imshow_components(labels))
cv.imshow('Labels2', testing.imshow_components(np.where(phraseLabels == 1, phraseLabels, 0)))
# cv.imshow('Texts', testing.imshow_components(textLabels))
# cv.imshow('Words', testing.imshow_components(wordLabels))
# cv.imshow('Phrases', testing.imshow_components(phraseLabels))
# cv.imshow('Non Texts', testing.imshow_components(nonTextLabels))
#
cv.waitKey()
cv.destroyAllWindows()

