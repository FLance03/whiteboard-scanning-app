import cv2 as cv
import numpy as np
import utils

# Problems:
#   - If contours with a lot of points are detected, program doesn't work
#   - Thresholding is kinda iffy and depends on the picture
#   - Fix warping
#   - Automate thresholding

cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 640, 80)
cv.createTrackbar("Threshold1", "Parameters", 75, 255, utils.empty)
cv.createTrackbar("Threshold2", "Parameters", 0, 255, utils.empty)

img = cv.imread('images/redundancyfrontcropped.jpg')
img2 = np.copy(img)

# Automate final image dimensions
heightImg = 640
widthImg = 1000

# test = canny_lines[0][0]
# cv.line(img2, (test[0], test[1]), (test[2], test[3]), (255,0,255), 3, cv.LINE_AA)


# Sobel edge detection with Hough Transform
# sobelx = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=3)
# abs_grad_x = cv.convertScaleAbs(sobelx)

# sobel_lines = cv.HoughLinesP(abs_grad_x, 1, np.pi / 180, 100, 1, 70, 80)

# if sobel_lines is not None:
#     for i in range(0, len(sobel_lines)):
#         l = sobel_lines[i][0]
#         cv.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

# cv.imshow("img", img)

# Display
res = img.copy()
while True:
    # For drawing contours on the display image
    imgContour = img.copy()

    # Preprocessing
    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), 0) # kernel size 5x5?

    # Canny edge detection with Hough Transform
    threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")
    canny = cv.Canny(blur, cv.CV_64F, threshold1, threshold2)
    kernel = np.ones((5,5))
    imgdil = cv.dilate(canny, kernel, iterations=1)
    bigContour = utils.getcontours(imgdil, imgContour)
    print(bigContour)
    if bigContour.size != 0:
        bigContour = utils.reorder(bigContour)
        print("Mypointsnew\n", bigContour)
        cv.drawContours(imgContour, bigContour, -1, (255,0,255), 20)
        imgContour = utils.drawRectangle(imgContour, bigContour, 2)
        pts1 = np.float32(bigContour)
        pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv.getPerspectiveTransform(pts1, pts2) 
        imgWarpColored = cv.warpPerspective(imgContour, matrix, (widthImg, heightImg))

    cv.imshow("rectangle", imgdil)
    cv.imshow("img", imgContour)
    cv.imshow("Contours", imgWarpColored)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break

    # canny_lines = cv.HoughLinesP(canny, 1, np.pi / 180, 70, None, 70, 20)

    # if canny_lines is not None:
    #     for i in range(0, len(canny_lines)):
    #         l = canny_lines[i][0]
    #         cv.line(img2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
