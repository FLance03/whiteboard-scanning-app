import cv2 as cv
from testing import testing

img = cv.imread('shapes.jpg')
imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(imgGrey, 240, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
testing.FullPrint(contours[0])
cnt = contours[0][:782, :]
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# for contour in contours:
#     approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
#     cv.drawContours(img, [approx], 0, (0, 0, 0), 5)
#     x, y, *_ = approx.ravel() # 0 for x-coordinate and 1 for y-coordinate
#     print(_)
#     if len(approx) == 3:
#         cv.putText(img, "Triangle", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 4:
#         x, y, w, h = cv.boundingRect(approx)
#         aspectRatio = w/h
#         print(aspectRatio)
#         if aspectRatio>=0.95 and aspectRatio<=1.05:
#             cv.putText(img, "Square", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#         else:
#             cv.putText(img, "Rectangle", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 5:
#         cv.putText(img, "Pentagon", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 10:
#         cv.putText(img, "Star", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     else:
#         cv.putText(img, "Circle", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
cv.imshow('shapes', img)
cv.waitKey()
cv.destroyAllWindows()