import cv2 as cv
import numpy as np
import math
from skimage.draw import line

# Threshold depends on image size and quality. If picture bugs try changing the threshold value
# For KB sized images 200 threshold is ideal, for MB 1000
# Width for the equation in plotting houghlines isn't static 1000, more research needed

def empty():
    pass

def resize(img):
    scale_percent = 15 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

def initRect(topline, botline):
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = topline[0]
    rect[1] = topline[1]
    rect[2] = botline[0]
    rect[3] = botline[1]

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

cv.namedWindow("Houghlines")
cv.resizeWindow("Houghlines", 900, 350)
# Threshold value here, change 800 to something else
cv.createTrackbar("threshold", "Houghlines", 800, 1000, empty)
cv.createTrackbar("lines", "Houghlines", 0, 1000, empty)
cv.createTrackbar("minLineLength", "Houghlines", 0, 1000, empty)
cv.createTrackbar("maxLineGap", "Houghlines", 0, 1000, empty)

img = cv.imread('images/redundancyright.jpg')
width = 5000
absWidth = 10000
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5,5), 0) 

# Create structuring elements
horizontal_size = 100
horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

# apply close to connect the white areas
morph = cv.morphologyEx(blur, cv.MORPH_OPEN, horizontalStructure)
morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, horizontalStructure)

# Sobel edge detection with Hough Transform
sobelx = cv.Sobel(morph, cv.CV_64F, 0, 1, ksize=3)
abs_grad_x = cv.convertScaleAbs(sobelx)
canny = cv.Canny(morph, cv.CV_64F, 75, 0)

while True:
    res = img.copy()
    wrp = img.copy()

    threshold = cv.getTrackbarPos("threshold", "Houghlines")
    lines = cv.getTrackbarPos("lines", "Houghlines")
    minLineLength = cv.getTrackbarPos("minLineLength", "Houghlines")
    maxLineGap = cv.getTrackbarPos("maxLineGap", "Houghlines")
    lines = cv.HoughLines(canny, 2, np.pi / 180, threshold, lines, minLineLength, maxLineGap)

    if lines is not None:
        halfline = int(img.shape[0]/2)
        botline = ((absWidth, absWidth), (absWidth, absWidth))
        topline = ((0,0), (0,0))
        # Might not need to compute for x-axis?
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + width*(-b)), int(y0 + width*(a)))
            pt2 = (int(x0 - width*(-b)), int(y0 - width*(a)))

            # Find actual points in x=0 and x=max-width
            print(pt1)
            (pt1, pt2) = findLinePoints(pt1, pt2, res.shape[1])

            # Choosing the top line and bottom line
            if pt1[1] < halfline and halfline - pt1[1] < halfline - topline[0][1]:
                topline = (pt1, pt2)
            elif pt1[1] > halfline and pt1[1] - halfline < botline[0][1] - halfline:
                botline = (pt1, pt2)

            # Displaying line coordinates
            print("-------------------------------")
            print("Line: ")
            print("PT1: ", pt1[0], pt1[1])
            print("PT2: ", pt2[0], pt2[1])

            cv.line(res, pt1, pt2, (0,0,255), 1, cv.LINE_AA)

    # top line blue, bot line green, halfline teal
    cv.line(res, topline[0], topline[1], (255,0,0), 5, cv.LINE_AA)
    cv.line(res, botline[0], botline[1], (0,255,0), 5, cv.LINE_AA)
    cv.line(res, (0, halfline), (width, halfline), (255,255,0), 5, cv.LINE_AA)

    # test line widths
    # cv.line(res, (0, 3000), (img.shape[1]-500, 3000), (0,255,255), 1, cv.LINE_AA)
    
    print("-------------------------------")
    print("-------------------------------")
    print("TopLine: ")
    print("PT1: ", topline[0])
    print("PT2: ", topline[1])
    print("-------------------------------")
    print("BotLine: ")
    print("PT1: ", botline[0])
    print("PT2: ", botline[1])
    print("-------------------------------")
    print("Halfline: ", halfline)

    # Warp Perspective and resize

    # I got this from interwebz, idk how to switch bl and br so that the warp doesn't break
    # am stuped help
    rect = initRect(topline, (botline[1], botline[0]))
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
    cv.imshow("sobelx", resize(abs_grad_x))
    cv.imshow("canny", resize(canny))
    cv.imshow("warped", resize(warped))

    if cv.waitKey(0) & 0xFF == ord('q'):
        break
