import testing

import numpy as np
import cv2 as cv

img = cv.imread('Screenshot (40).png')
# retImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
# for r, row in enumerate(img):
#     for c, col in enumerate(row):
#         if np.all(col == [0, 255, 135]):
#             retImg[max(0, r-5):min(len(retImg), r+5), max(0, c-5):min(len(retImg[0]), c+5)] = 255
# img = retImg
testing.PlotIt(img)