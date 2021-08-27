import cv2 as cv
import numpy as np

def empty(a):
    pass

# Draws contours and returns the biggest
def getcontours(img, imgContour):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    biggestContour = np.array([])
    biggestA = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000 and area > biggestA : # Change 1000 to a formula that's around 50% of the size of the picture
            cv.drawContours(imgContour, cnt, -1, (255,0,255), 2)
            biggestA = area
            peri = cv.arcLength(cnt, True)
            biggestContour = approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            # print(approx)
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(imgContour, "Points: "+ str(len(approx)), (x, y+20), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 1)
            cv.putText(imgContour, "Area: "+ str(int(area)), (x, y+40), cv.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 1)
    
    return biggestContour

def reorder(myPoints):
    # print(myPoints)
    length = myPoints.size
    myPoints = myPoints.reshape((length//2, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def drawRectangle(img, biggest, thickness):
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (255, 0, 255), 3)
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (255, 0, 255), 3)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (255, 0, 255), 3)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (255, 0, 255), 3)

    return img
