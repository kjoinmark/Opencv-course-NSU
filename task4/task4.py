import cv2 as cv
import numpy as np

img = cv.imread("lena.png", cv.CV_32F)
# 1 ручная коррекция
alpha = 1.5
beta = 20
img2 = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
cv.imwrite("correct1.jpg", img2)

# 2 гамма коррекция
gamma = 2
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res = cv.LUT(img, lookUpTable)

cv.imwrite("correct2.jpg", res)

from matplotlib import pyplot as plt

# 3 гистограмма
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr)

plt.savefig("hist.jpg")


# plt.hist(img)

# 4 гистограмма
def hisEqulColor(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    return img


img_eq = hisEqulColor(img)
cv.imwrite('img_eq.jpg', img_eq)

for i, col in enumerate(color):
    histr = cv.calcHist([img_eq], [i], None, [256], [0, 256])
    plt.plot(histr)

plt.savefig("hist_eq.jpg")
