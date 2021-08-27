import cv2 as cv
import numpy as np
import math
import utils

cv.namedWindow("Houghlines")
cv.resizeWindow("Houghlines", 900, 500)
cv.createTrackbar("threshold", "Houghlines", 165, 1000, utils.empty)
cv.createTrackbar("lines", "Houghlines", 0, 1000, utils.empty)
cv.createTrackbar("minLineLength", "Houghlines", 0, 1000, utils.empty)
cv.createTrackbar("maxLineGap", "Houghlines", 0, 1000, utils.empty)

img = cv.imread('images/board2.jpg')

# Sobel edge detection with Hough Transform
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5,5), 0) 

# Create structuring elements
# horizontal_size = 100
# horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

# apply close to connect the white areas
# morph = cv.morphologyEx(blur, cv.MORPH_OPEN, horizontalStructure)
# morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, horizontalStructure)

sobelx = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=3)
abs_grad_x = cv.convertScaleAbs(sobelx)
canny = cv.Canny(abs_grad_x, cv.CV_64F, 75, 0)

# Display
cv.imshow("res", img)
cv.imshow("sobelx", abs_grad_x)
cv.imshow("canny", canny)

while True:
#     res = img.copy()

#     threshold = cv.getTrackbarPos("threshold", "Houghlines")
#     lines = cv.getTrackbarPos("lines", "Houghlines")
#     minLineLength = cv.getTrackbarPos("minLineLength", "Houghlines")
#     maxLineGap = cv.getTrackbarPos("maxLineGap", "Houghlines")
#     lines = cv.HoughLines(canny, 1, np.pi / 180, threshold, lines, minLineLength, maxLineGap)

#     if lines is not None:
#         for i in range(0, len(lines)):
#             rho = lines[i][0][0]
#             theta = lines[i][0][1]
#             a = math.cos(theta)
#             b = math.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#             cv.line(res, pt1, pt2, (0,0,255), 1, cv.LINE_AA)

    if cv.waitKey(0) & 0xFF == ord('q'):
        break
