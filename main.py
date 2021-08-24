import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from Steps import (SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5)
from testing import testing

img = cv.imread('./testing/pics and texts/iotbinarized.jpg', 0)
labels, labelsInfo, textNonText, textLabels, wordLabels, phraseLabels, nonTextLabels = Step5.main(img)


cv.imshow('Labels', testing.imshow_components(labels))
cv.imshow('Texts', testing.imshow_components(textLabels))
cv.imshow('Words', testing.imshow_components(wordLabels))
cv.imshow('Phrases', testing.imshow_components(phraseLabels))
cv.imshow('Non Texts', testing.imshow_components(nonTextLabels))

cv.waitKey()
cv.destroyAllWindows()

