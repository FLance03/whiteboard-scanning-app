from random import sample
from pprint import pprint as p
from time import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from testing import testing

start = time()


def GetPlane(points):
    # np.asarray does not copy iff its a compatible ndarray
    point1, point2, point3 = [np.asarray(point) for point in points]
    pq, qr = point2-point1, point3-point1
    a, b, c = np.cross(pq, qr)
    d = - (a*point1[0] + b*point1[1] + c*point1[2])
    # The research based upon used the equation ax+by-z+c=0 with a 3D point (x,y,z). Thus, c forced to be -1 (For readability)
    a, b, d, c = -a/c, -b/c, -d/c, -1
    # Did not return c anymore
    return a, b, d


def ComputeMedianError(cells, random):
    points = np.array([cells[i] for i in random])
    a, b, d = GetPlane(points)
    # Matrix multiply rows of points with (a, b, d) parameters.
    errors = (cells.dot((a, b, -1)) + d)**2
    return np.median(errors), a, b, d, random


def SplitIntoCells(img, width=15, height=15):
    # Returns a 2D list of ndarrays where each ndarray contains the usual 3D opencv data for an image
    cells = []
    for x in range(0,img.shape[0],height):
        cells.append([])
        for y in range(0,img.shape[1],width):
            endHeight = x+height
            endWidth = y+width
            if endHeight > img.shape[0]:
                endHeight = img.shape[0]
            if endWidth > img.shape[1]:
                endWidth = img.shape[1]
            cells[-1].append(np.array(img[x:endHeight,y:endWidth]))
    return cells


def FixOutliers(top25LumiCells):
    numCells = len(top25LumiCells) * len(top25LumiCells[0])
    # Separate color channel since top25LumiCells is a 2d array of bgr data. A point (x, y, z) would be (column, row, intensity) of that cell
    # List (3 elements) of np arrays of np arrays forming a 3 element list of matrices where each row of a matrix is a point of (column, row, intensity)
    colorsPoints = [np.array([(c, r, top25LumiCells[r][c][i]) for r in range(len(top25LumiCells)) for c in range(len(top25LumiCells[r]))]) for i in range(3)]
    for i in range(3): # Per color channel
        randoms = []
        while len(randoms) < 35:
            # Make sure that the x and y of the three points chosen in a sample are not collinear
            p1, p2, p3 = sample(range(0, numCells), 3)
            points = colorsPoints[i]
            # Continue if:
            #   1) The x-coordinates of the three points are equal or,
            #   2) The y-coordinates of the three points are equal or,
            #   3) The slope of the line from first and second points is equal to the slope of the line from the first and third points
            #       (together with the common first point, this would imply collinearity)
            if (points[p2][0]-points[p1][0])*(points[p3][1]-points[p1][1]) != (points[p2][1]-points[p1][1])*(points[p3][0]-points[p1][0]):
                randoms.append([p1, p2, p3])
        medianErrors = [ComputeMedianError(colorsPoints[i], random) for random in randoms]
        # medianErrors is a list of (median, a, b, d) where a, b, d are the plane parameters
        # Finding the tuple with the least median
        minMedianError = medianErrors[0]
        for j in range(1, 35):  # For each of the 35 random points...
            if medianErrors[j][0] < minMedianError[0]:
                minMedianError = medianErrors[j]
        # Formula used by the research article. Above such is considered an outlier
        outlierThresh = 2.5 * 1.4826 * minMedianError[0]
        errors = colorsPoints[i].dot((minMedianError[1], minMedianError[2], -1)) + minMedianError[3]
        # np.where return a tuple of numpy arrays, one for each dimension of the input
        checkOutliers = abs(errors) > outlierThresh
        outlierIndices, nonOutlierIndices = [], []
        for index, isOutlier in enumerate(checkOutliers):
            outlierIndices.append(index) if isOutlier else nonOutlierIndices.append(index)
        # In the research based on, the formula to get the least squares solution is p = (A^T A)^-1 A^T z
        # Where A is a matrix where each row is (xi, yi, 1) and z is a column matrix of z1, z2, ..., zn
        # z below is a row matrix
            # A = np.insert(colorsPoints[i][nonOutlierIndices][:, :-1], 2, 1, axis=1)
            # z = colorsPoints[i][nonOutlierIndices][:, -1]
            # p = (np.linalg.pinv(A.transpose() @ A) @ A.transpose()).dot(z)
        A = np.insert(colorsPoints[i][nonOutlierIndices][:, :-1], 2, 1, axis=1)
        z = colorsPoints[i][nonOutlierIndices][:, -1]
        a, b, c = np.linalg.lstsq(A, z, rcond=1)[0]
        # Get the new colors for the outlier cells based on the least squares parameters (a, b, c above)
        outlierPos = colorsPoints[i][outlierIndices][:, :-1]
        outlierNewVals = outlierPos.dot((a, b)) + c
        print('Number of Outliers for color channel ' + ('blue', 'green', 'red')[i] + ': ' + str(len(outlierPos)))
        for index, outlier in enumerate(outlierPos):
            top25LumiCells[outlier[1]][outlier[0]][i] = outlierNewVals[index]


def main(img, CELLGROUPSIZE = 70, CELLSIZE = 15):
    cells = SplitIntoCells(img, width=CELLSIZE, height=CELLSIZE)
    top25LumiCells = []
    for cellRow in cells:
        top25LumiCells.append([])
        for cell in cellRow:
            # Reshape to "ravel" since difference between height and width does not matter.
            # Partition at the 75% then get the mean from the top 75%
            flatCell = cell.reshape(-1,3)
            topQuartile = len(flatCell) - len(flatCell)//4
            top25LumiCells[-1].append(np.round(np.mean(np.partition(flatCell, topQuartile, axis=0)[topQuartile:], axis=0)).astype(np.uint8))
    if len(top25LumiCells) * len(top25LumiCells[0]) >= 3:
        for x in range(0, len(top25LumiCells), CELLGROUPSIZE):
            for y in range(0, len(top25LumiCells[x]), CELLGROUPSIZE):
                endHeight = x + CELLGROUPSIZE
                endWidth = y + CELLGROUPSIZE
                # When the remaining number of cells is less than CELLGROUPSIZE in height or width
                if endHeight > len(top25LumiCells):
                    endHeight = len(top25LumiCells)
                    # If there is a lack of cells then backtrack the start (variable x) to still include CELLGROUPSIZE cells in the current iteration
                    # If statement incase CELLGROUPSIZE is large relative to the length of top25LumiCells which would go negative
                    x = len(top25LumiCells) - CELLGROUPSIZE if len(top25LumiCells) - CELLGROUPSIZE > 0 else 0
                if endWidth > len(top25LumiCells[x]):
                    endWidth = len(top25LumiCells[x])
                    y = len(top25LumiCells[x]) - CELLGROUPSIZE if len(top25LumiCells[x]) - CELLGROUPSIZE > 0 else 0
                # Tried to imitate FixOutliers(top25LumiCells[x:endHeight,y:endWidth]) since top25LumiCells is a list of lists not ndarray
                interestSquare = [[j for j in top25LumiCells[i][y:endWidth]] for i in range(x, endHeight)]
                FixOutliers(interestSquare)
    # Based on the representative colors from top25LumiCells, scale the color for each cell (refer to research article for the formulas being based)
    for row in range(len(top25LumiCells)):
        for col in range(len(top25LumiCells[row])):
            whiteboardColor = top25LumiCells[row][col]
            cell = cells[row][col]
            for index, colorChannelIntensity in enumerate(whiteboardColor):
                # For each color channel, change intensity to min(1, color/whiteboard color) unless division by zero then default to 1
                if colorChannelIntensity == 0:
                    cell[:, :, index] = 255
                else:
                    # Additional variable out since np.minimum gives float so assigning to some slice of cell with type uint cuts it to (mostly) 0
                    # Also preferred this where instead of making the entire cell float, only the temp variable, out, is a floar
                    out = np.minimum(1, cell[:, :, index]/colorChannelIntensity)
                    cell[:, :, index] = np.round((0.5 - 0.5 * np.cos(out ** 2 * np.pi)) * 255).astype(np.uint8)
            cells[row][col] = cell
    # Merge/Concatenate all cells back to the complete image
    merged = []
    for cellRow in cells:
        mergedRow = []
        for cell in cellRow:
            mergedRow.append(cell)
        merged.append(cv.hconcat(mergedRow))
    merged = cv.vconcat(merged)

    print('Step 2 Time taken: ' + str(time()-start))
    return merged

    cv.imshow('Original', testing.ResizeWithAspectRatio(img, height=500))
    # cv.imshow('Modified', testing.ResizeWithAspectRatio(merged, height=500))

    # gray1 = cv.cvtColor(merged, cv.COLOR_BGR2GRAY)
    # _, otsu1 = cv.threshold(gray1, 0, 255, cv.THRESH_OTSU)
    # cv.imshow('Otsu', otsu1)
    #
    # gray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # _, otsu3 = cv.threshold(gray2, 0, 255, cv.THRESH_OTSU)
    # cv.imshow('Without Step 2 Otsu', otsu3)

    cv.waitKey()
    # cv.destroyAllWindows()
# main(cv.imread('lefthard.jpg'))

