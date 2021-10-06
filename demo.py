import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from Steps import (SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5,
                   FeatureExtraction6 as Step6)
from testing import testing

testImages = ['1fcropped', '2fcropped', '3fcropped']
names = ['first', 'second', 'third']
# testImages = ['iot']
# names = ['iot']
anded = []
for ind, testImage in enumerate(testImages):
    img = cv.imread('./testing/pics and texts/Group Tests/1/' + testImage + '.jpg')
    assert img is not None
    img = Step2.main(img, CELLGROUPSIZE = 50, CELLSIZE = 5)
    cv.imshow(names[ind] + ': Saturation', img)
    cv.waitKey()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, dsize=(500, 500), interpolation=cv.INTER_LINEAR)
    _, bw = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    bw = bw[15:-15]
    anded.append(np.where(bw == 0, 1, 0))
    cv.imshow(names[ind] + ': Binarization', bw)
    cv.waitKey()

    # Remove the possible lines from the blackboard edges
    labels, labelsInfo, textLabels, wordLabels, phraseLabels, nonTextLabels = Step5.main(bw)
    cv.imshow(names[ind] + ': Texts', testing.ResizeWithAspectRatio(testing.imshow_components(wordLabels), width=455))
    cv.imshow(names[ind] + ': Non-Texts', testing.ResizeWithAspectRatio(testing.imshow_components(nonTextLabels), width=455))
    cv.waitKey()
    cv.destroyAllWindows()



