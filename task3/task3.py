import cv2 as cv

img = cv.imread("text.jpg", cv.IMREAD_GRAYSCALE)

img2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                            cv.THRESH_BINARY, 11, 2)

cv.imwrite("result.jpg", img2)
