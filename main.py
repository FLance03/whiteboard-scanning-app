import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from Steps import SaturationAndOutlier2 as Step2
from testing import testing

img = cv.imread('./testing/iot.jpg')
img = Step2.main(img, CELLGROUPSIZE=20)

cv.imshow('Image Step 2', img)
# Step 3: Binarization
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


_, otsu = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
# Remove the possible lines from the blackboard edges
img = otsu[15:-15]
cv.imshow('Otsu', otsu)
# Step 4: Closing (removing black) then Opening (removing white)
# For now nothing sah cause image had alot of 1 pixel thick lines
# plt.subplot(1, 2, 1)
# plt.imshow(img, 'gray')
# plt.title('Image')
#
# kernel = np.ones((2, 1), np.uint8)
# img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
# img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# plt.subplot(1, 2, 2)
# plt.imshow(img, 'gray')
# plt.title('Close')
# plt.show()

# Step 5 Component Grouping
# img = ConnectedComponentGrouping5(img)
cv.imwrite('iotbinarized.jpg', img)
cv.imshow('Close', img)

cv.waitKey()
cv.destroyAllWindows()

