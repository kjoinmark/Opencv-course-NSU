import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
Реализовать сегментацию изображений с помощью алгоритма watershed (потестировать на различных примерах)  
"""

img = cv.imread('cards.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

plt.imshow(thresh)
plt.show()

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=4)
plt.imshow(sure_bg)
plt.show()
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)

plt.imshow(markers)
plt.show()

img[markers == -1] = [255, 0, 0]

cv.imwrite("result.jpg", img)
