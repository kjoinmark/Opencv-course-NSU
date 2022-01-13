import cv2 as cv
import numpy as np

"""
Реализовать сегментацию изображений с помощью алгоритма graphcut (потестировать на различных примерах, при различных параметрах).
"""
img = cv.imread('messi.jpg', cv.IMREAD_UNCHANGED)
start = img.copy()

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (450, 0, 300, 675)

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

kernel = np.ones((3, 3), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

cv.imwrite('result.jpg', img)
