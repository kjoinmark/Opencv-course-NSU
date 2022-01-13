import cv2 as cv
import numpy as np

img = cv.imread("lena.png", 0)
rows, cols = np.shape(img)

noise_range = 30

# 2 шума: равномерный шум и Гауссов

uniform_noise = np.zeros((rows, cols), dtype=np.uint8)
cv.randu(uniform_noise, 0, noise_range)
uniform_noise = uniform_noise + img

gauss_noise = np.zeros((rows, cols), dtype=np.uint8)
cv.randn(gauss_noise, 0, noise_range)
gauss_noise = gauss_noise + img

cv.imshow("uniform", uniform_noise)
cv.imshow("gaussian", gauss_noise)

cv.waitKey()

# Гауссов фильтр и медианный
k = 5
gfilter_g = cv.blur(gauss_noise, (k, k))
gfilter_u = cv.blur(uniform_noise, (k, k))

cv.imwrite("1gaussian_gauss.jpg", gfilter_g)
cv.imwrite("1uniform_gauss.jpg", gfilter_u)

cv.waitKey()

median_g = cv.medianBlur(gauss_noise, k)
median_u = cv.medianBlur(uniform_noise, k)

cv.imwrite("2gaussian_median.jpg", median_g)
cv.imwrite("2uniform_median.jpg", median_u)

cv.waitKey()

# Фильтрация маской

kernel = np.ones((5, 5), np.float32) / 25
mask_g = cv.filter2D(gauss_noise, -1, kernel)
mask_u = cv.filter2D(uniform_noise, -1, kernel)

cv.imwrite("3gaussian_mask.jpg", mask_g)
cv.imwrite("3uniform_mask.jpg", mask_u)

# Собель

kernelsize = np.int(5)

sobelx_g = cv.Sobel(gauss_noise, cv.CV_64F, 1, 0, kernelsize)
sobely_g = cv.Sobel(gauss_noise, cv.CV_64F, 0, 1, kernelsize)

sobelx_u = cv.Sobel(uniform_noise, cv.CV_64F, 1, 0, kernelsize)
sobely_u = cv.Sobel(uniform_noise, cv.CV_64F, 0, 1, kernelsize)

cv.imwrite("4gaussian_sobelx.jpg", sobelx_g)
cv.imwrite("4gaussian_sobely.jpg", sobely_g)
cv.imwrite("4uniform_sobelx.jpg", sobelx_u)
cv.imwrite("4uniform_sobely.jpg", sobely_u)


# Лаплас

def laplas(img, kernelsize, ddepth):
    dst = cv.Laplacian(img, ddepth, kernelsize)
    return cv.convertScaleAbs(dst)


kernelsize = np.int(5)
ddepth = cv.CV_16S

laplas_g = laplas(gauss_noise, kernelsize, ddepth)
laplas_u = laplas(gauss_noise, kernelsize, ddepth)

cv.imwrite("5gaussian_laplas.jpg", sobely_g)
cv.imwrite("5uniform_laplas.jpg", sobelx_u)
