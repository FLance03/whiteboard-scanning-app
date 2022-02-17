from xml.etree.ElementTree import tostring
import cv2 as cv
import numpy as np
import math

def empty():
    pass

def resize(img):
    scale_percent =  80 # percent of original size
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

def Preprocessing1(img):
    # Values for Houghlines here
    width = img.shape[1]
    height = img.shape[0]
    # Really need to find a formula for threshold
    if height >= 1080 or width >= 1920:
        threshold = 900
    elif height >= 700 or width >= 1200:
        threshold = 800
    elif height >= 500 or width >= 500:
        threshold = 600
    else:
        threshold = 400
    lines_number = 0
    minLineLength = 0
    maxLineGap = 0
    absWidth = 10000

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (7,7), 0) 

    # Create structuring elements
    cols = width
    horizontal_size = cols // 30
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

    # apply close to connect the white areas
    morph = cv.morphologyEx(blur, cv.MORPH_OPEN, horizontalStructure)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, horizontalStructure)

    # Sobel edge detection with Hough Transform
    sobelx = cv.Sobel(morph, cv.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv.convertScaleAbs(sobelx)
    canny = cv.Canny(abs_grad_x, cv.CV_64F, 75, 0)
    # cv.imshow("canny", resize(canny))

    res = img.copy()
    wrp = img.copy()

    lines = cv.HoughLines(canny, 2, np.pi / 180, threshold, lines_number, minLineLength, maxLineGap)

    # Default values
    halfline = int(height/2)
    botline = ((absWidth, absWidth), (absWidth, absWidth))
    topline = ((0,0), (0,0))

    if lines is not None:
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
            (pt1, pt2) = findLinePoints(pt1, pt2, res.shape[1])

            # Choosing the top line and bottom line
            # Instead of comparing y[0], compare middle point y of the line??

            if pt1[1] < halfline and halfline - pt1[1] < halfline - topline[0][1]:
                print('new topline:', topline)
                topline = (pt1, pt2)
            elif pt1[1] > halfline and pt1[1] - halfline < botline[0][1] - halfline:
                print('new botline:', botline)
                botline = (pt1, pt2)

            # if pt1[1] < halfline:
            #     topline = (pt1, pt2)
            # elif pt1[1] > halfline:
            #     botline = (pt1, pt2)

            # Displaying line coordinates
            print("-------------------------------")
            print("Line: ")
            print("PT1: ", pt1[0], pt1[1])
            print("PT2: ", pt2[0], pt2[1])

            cv.line(res, pt1, pt2, (0,0,255), 1, cv.LINE_AA)
    else:
        print('ERROR: Wasnt able to detect lines after Preprocessing. Change threshold.')
        botline = ((0, height), (width, height))
        topline = ((0, 0), (width, 0))

    # Might not need to compute for x-axis?
    # if only topline is detected, then calculate botline and vice versa
    if botline == ((absWidth, absWidth), (absWidth, absWidth)) and topline != ((0,0), (0,0)):
        print('no botline detected')
        botline = ((topline[0][0], height-topline[0][1]), (topline[1][0], height-topline[1][1]))
    elif topline == ((0,0), (0,0)) and botline != ((absWidth, absWidth), (absWidth, absWidth)):
        print('no topline detected')
        topline = ((botline[0][0], height-botline[0][1]), (botline[1][0], height-botline[1][1]))

    # top line blue, bot line green, halfline teal
    # cv.imshow("lines detected", resize(res))
    cv.line(res, topline[0], topline[1], (255,0,0), 5, cv.LINE_AA)
    cv.line(res, botline[0], botline[1], (0,255,0), 5, cv.LINE_AA)
    cv.line(res, (0, halfline), (width, halfline), (255,255,0), 5, cv.LINE_AA)
    # cv.imshow("topline botline", resize(res))
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

    rect = initRect(topline, (botline[1], botline[0]))
    (tl, tr, br, bl) = rect
    print("Rect:", rect)

    # Maximum width computation
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    print("Width: ", width)
    print("MaxWidth: ", maxWidth)

    # Maximum height computation
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    
    # Computing the perspective transform matrix + application
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(wrp, M, (maxWidth, maxHeight))
    print('threshold: ', threshold)

    # Display
    # cv.imshow("res", resize(res))
    # cv.imshow("morph", resize(morph))
    # cv.imshow("sobelx", resize(abs_grad_x))
    # cv.imshow("canny", resize(canny))
    # cv.imshow("warped", resize(warped))

    return warped

    # cv.imwrite(imageNameOutput+'.jpg',resize(warped))

# test 
# img = cv.imread('2.png')
# retval = Preprocessing1(img)
# cv.imshow('retval', resize(retval))
# cv.waitKey()


# test for many
# for i in range(3, 7):
#     img = cv.imread(str(i)+'.png')
#     retval = Preprocessing1(img)
#     cv.imshow(str(i), resize(retval))
#     cv.waitKey()
#     cv.destroyAllWindows()