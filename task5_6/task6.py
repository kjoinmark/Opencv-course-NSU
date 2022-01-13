import cv2 as cv
import numpy as np

img = cv.imread('out0_0.jpg')
rows, cols, chan = np.shape(img)
print(rows, cols)
ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
channels = cv.split(ycrcb)
edges = np.zeros(np.shape(channels[0]), np.uint8)

for channel in channels:
    ch = cv.blur(channel, (5, 5))
    canny = cv.Canny(ch, 100, 200, 5)
    edges += canny

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
dilated = cv.dilate(edges, kernel)
contours, hierarchy = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cntr = np.zeros((0, 1, 2), np.uint8)

min = (rows + cols) * 0.1
print(min)

for i, contour in enumerate(contours):
    length = cv.arcLength(contour, True)
    if (length > min):
        print("a")
        cntr = np.vstack((cntr, contour))

print(np.shape(cntr))
hull = cv.convexHull(cntr)
mask = np.zeros((rows, cols), np.uint8)

cv.fillConvexPoly(mask, hull, color=255)

main_contour, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(np.shape(main_contour[0]))
area = cv.contourArea(main_contour[0])
print(area)
print(int(area / (rows * cols) * 100), '%')


def define_size(area1, area2):
    per = int(area1 / area2 * 100)
    if (per > 50):
        print("VERY BIG")
        return 0
    if (per > 20):
        print("BIG")
        return 0
    if (per > 5):
        print("MEDIUM")
        return 0
    print("SMALL")
    return 0


define_size(area, rows * cols)


def define_shape(contour):
    area = cv.contourArea(contour)
    (x, y), radius = cv.minEnclosingCircle(contour)
    # center = (int(x), int(y))

    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    retval, triangle = cv.minEnclosingTriangle(contour)

    area_arr = list()
    area_arr.append(np.pi * radius ** 2)
    area_arr.append(cv.contourArea(box))
    area_arr.append(cv.contourArea(triangle))
    min_index = np.argmin(area_arr)

    if min_index == 0:
        print("It might be circle")
    if min_index == 1:
        print("It might be rectangle")
    if min_index == 2:
        print("It might be triangle")
    # cv.circle(mask, center, int(radius), 100, 2)
    # cv.drawContours(mask, [box], 0, 50, 2)


define_shape(main_contour[0])

cv.imshow("original", img)
cv.imshow("mask", mask)
cv.imwrite("mask.jpg", mask)

cv.waitKey()
