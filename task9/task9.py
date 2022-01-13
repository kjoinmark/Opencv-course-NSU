import cv2 as cv
import numpy as np

img_rgb = cv.imread('sudoku.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
ch = cv.medianBlur(img_gray, 5)
dst = cv.Canny(ch, 100, 200, 5)
cv.imshow("canny", dst)
cv.waitKey()

# Преобразование Хафа

# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(cdst, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)

linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv.LINE_AA)

cv.imshow("Source", img_gray)
cv.imshow("Standard Hough Line Transform", cdst)
cv.imshow("Probabilistic Line Transform", cdstP)
cv.waitKey()

# Преобразование Хафа для кругов

img = cv.imread('circles.jpg', 0)
cimg = img.copy()
ch_c = cv.medianBlur(cimg, 5)
dst_c = cv.Canny(ch_c, 100, 200, 5)

cv.imshow("canny", dst_c)
cv.waitKey()

rows = cimg.shape[0]
circles = cv.HoughCircles(dst_c, cv.HOUGH_GRADIENT, 1, rows / 8,
                          param1=100, param2=30,
                          minRadius=50, maxRadius=120)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(img, center, radius, (255, 0, 255), 3)

cv.imshow('detected circles', img)
cv.waitKey(0)
