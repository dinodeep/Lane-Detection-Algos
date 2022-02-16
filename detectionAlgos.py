import cv2 as cv
import numpy as np
import math

lineColor = (243, 200, 80)
fillColor = (245, 220, 180)

def mergeLines(lines, thetadiff=0.00):
    # average out lines that are going in directions
    pairs = []
    for i in range(len(lines)):
        l = lines[i,0]
        pairs.append((l, math.atan((l[3] - l[1]) / (l[2] - l[0]))))

    leftLine = np.array([0, 0, 0, 0])
    rightLine = np.array([0, 0, 0, 0])
    lCount, rCount = 0, 0
    for l, slope in pairs:
        # average the points
        if slope < 0:
            leftLine += l
            lCount += 1
        else:
            rightLine += l
            rCount += 1

    leftLine = leftLine // lCount
    rightLine = rightLine // rCount

    return [rightLine, leftLine]

def boundLines(lines, minY, maxY):
    # bound lines using point-slope formula and quick maths
    
    extendedLines = []
    for l in lines:
        x0, y0, x1, y1 = l
        slope = math.atan((y1-y0)/(x1-x0))

        # extend line to bottom area
        maxX = int(((maxY - y0) / slope) + x0)
        if y0 > y1:
            x0, y0, x1, y1 = [x1, y1, maxX, maxY]
        else:
            x0, y0, x1, y1 = [x0, y0, maxX, maxY]

        # cutoff the top of the line
        minX = int(((minY - y0) / slope) + x0)
        if y0 < minY:
            x0 = minX
            y0 = minY

        extendedLines.append([x0, y0, x1, y1])
        

    return extendedLines

    


def method0(img):
    # method obtained from: https://medium.com/@techreigns/how-does-a-self-driving-car-detect-lanes-on-the-road-fc516c789984

    # convert image to grayscale - prep for Canny edge detection
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", grayImg)

    # apply Gaussian blur - remove unnecessary noise
    blurredImg = cv.GaussianBlur(grayImg, (5,5), 1)
    # cv.imshow("blurred", blurredImg)

    # view Sobel values because they are cool and swanky
    # sobelx = cv.Sobel(src=blurredImg, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
    # sobely = cv.Sobel(src=blurredImg, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
    # sobelxy = cv.Sobel(src=blurredImg, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    # cv.imshow("sobelx", sobelx)
    # cv.imshow("sobely", sobely)
    # cv.imshow("sobelxy", sobelxy)

    # perform Canny edge detection - get edge detection
    edges = cv.Canny(blurredImg, 150, 255)
    # cv.imshow("Canny Edge Detection", edges)

    # create trapezoidal region of interest - don't want lines from off of road
    mask = np.zeros(grayImg.shape)
    nrows = mask.shape[0]
    ncols = mask.shape[1]
    # for some reason: the rows and columns are swapped which makes things kinds of weird
    vertices = np.array([[(int(ncols//2 - 20),int(nrows//2) + 20), (0,nrows), (ncols,nrows), (int(ncols//2 + 20),int(nrows//2)  + 20)]], dtype=np.int32)
    print(vertices.shape)
    mask = cv.fillPoly(mask, vertices, 255)
    maskedEdges = cv.bitwise_and(edges.astype(np.uint8), mask.astype(np.uint8))
    # cv.imshow("masked image", maskedEdges)

    # detect for lines on the region of interest using Hough Transforms
    rho = 10            # distance resolution in pixels of the Hough grid
    theta = np.pi / 360 # angular resolution in radians of the Hough grid
    threshold = 100      # the minimum number of votes (intersections in the Hough grid cell)
    minLineLength = 50  # min number of pixels making up a line
    maxLineGap = 1000     # max gap in pixels between connectable line segments
    linedImage = np.copy(img)
    linesP = cv.HoughLinesP(maskedEdges, rho, theta, threshold, None, minLineLength, maxLineGap)

    # filter out similar lines
    lines = mergeLines(linesP)
    rl, ll = boundLines(lines, nrows // 2 + 40, nrows - 30)
    cv.line(linedImage, (ll[0], ll[1]), (ll[2], ll[3]), lineColor, 3, cv.LINE_AA)
    cv.line(linedImage, (rl[0], rl[1]), (rl[2], rl[3]), lineColor, 3, cv.LINE_AA)

    # create semi-transparent fill between the lane lines (ty to https://gist.github.com/IAmSuyogJadhav/305bfd9a0605a4c096383408bee7fd5c)
    alpha = 0.5
    vertices = np.array([[(ll[2], ll[3]), (ll[0], ll[1]), (rl[0], rl[1]), (rl[2], rl[3])]])
    overlay = np.copy(linedImage)
    overlay = cv.fillPoly(overlay, vertices, fillColor)
    linedImage = cv.addWeighted(overlay, alpha, linedImage, 1 - alpha, -1)

    cv.imshow("Lined image", linedImage)


    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return linedImage