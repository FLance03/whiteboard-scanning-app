import cv2 as cv
import numpy as np
import math

import testing as testing

# Note:
# - Change these values to change the coordinates for topline points and botline points

# - Change HERE!!!!!
topLpt = (0, 480)
topRpt = (2985, 1115)
botLpt = (0, 5055)
botRpt = (2985, 4167)
<<<<<<< HEAD
imagename = "step1redundancyfront"
=======
imageNameInput = "redundancyleft"
imageNameOutput = "step1redundancyfront"
>>>>>>> b9f455bcdebc9467c61bf9102e7274d211648f29

def empty():
    pass

def resize(img):
    scale_percent = 15 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

def initRect():
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = topLpt
    rect[1] = topRpt
    rect[2] = botRpt
    rect[3] = botLpt

    return rect

def findLinePoints(p1, p2, width):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    if(x1>x2):
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    print(x1, y1, x2, y2)
    slope = (y2-y1)/(x2-x1)

    # Solve for first pair of coordinates
    x2 = 0
    y2 = int(slope*(x2-x1)+y1)

    # Rearrange
    x1, y1 = int(x2), int(y2)

    # Solve for second pair of coordinates
    x2 = int(width)
    y2 = int(slope*(x2-x1)+y1)

    return ((x1, y1), (x2, y2))

# cv.namedWindow("Houghlines")
# cv.resizeWindow("Houghlines", 900, 350)
# # Threshold value here, change 800 to something else
# cv.createTrackbar("threshold", "Houghlines", 800, 10000, empty)
# cv.createTrackbar("lines", "Houghlines", 0, 1000, empty)
# cv.createTrackbar("minLineLength", "Houghlines", 0, 1000, empty)
# cv.createTrackbar("maxLineGap", "Houghlines", 0, 1000, empty)

img = cv.imread('images/'+imageNameInput+'.jpg')
width = 5000
absWidth = 10000
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blur = cv.GaussianBlur(gray, (5,5), 0) 

# # Create structuring elements
# horizontal_size = 100
# horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

# # apply close to connect the white areas
# morph = cv.morphologyEx(blur, cv.MORPH_OPEN, horizontalStructure)
# morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, horizontalStructure)

# # Sobel edge detection with Hough Transform
# sobelx = cv.Sobel(morph, cv.CV_64F, 0, 1, ksize=3)
# abs_grad_x = cv.convertScaleAbs(sobelx)
# canny = cv.Canny(morph, cv.CV_64F, 75, 0)

while True:
    res = img.copy()
    wrp = img.copy()

    halfline = int(img.shape[0]/2)

    # top line blue, bot line green, halfline teal
    cv.line(res, topLpt, topRpt, (255,0,0), 5, cv.LINE_AA)
    cv.line(res, botLpt, botRpt, (0,255,0), 5, cv.LINE_AA)
    cv.line(res, (0, halfline), (width, halfline), (255,255,0), 5, cv.LINE_AA)

    # test line widths
    # cv.line(res, (0, 3000), (img.shape[1]-500, 3000), (0,255,255), 1, cv.LINE_AA)
    
    print("-------------------------------")
    print("-------------------------------")
    print("TopLine: ")
    print("PT1: ", topLpt)
    print("PT2: ", topRpt)
    print("-------------------------------")
    print("BotLine: ")
    print("PT1: ", botLpt)
    print("PT2: ", botRpt)
    print("-------------------------------")
    print("Halfline: ", halfline)

    # Warp Perspective and resize
    rect = initRect()
    (tl, tr, br, bl) = rect
    print("Rect:", rect)

    # Maximum width computation
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Maximum height computation
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
		[0, 0],
		[halfline - 1, 0],
		[halfline - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    
    # Computing the perspective transform matrix + application
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(wrp, M, (halfline, maxHeight))

    # Display
    cv.imshow("res", resize(res))
    # cv.imshow("morph", resize(morph))
    cv.imshow("warped", resize(warped))
    cv.imwrite(imageNameOutput+'.jpg',resize(warped))
    if cv.waitKey(0) & 0xFF == ord('q'):
        break
