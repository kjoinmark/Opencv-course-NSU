import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
Использовать детектор углов Харриса и детектор особых точек Ши-Томаси, исследовать влияние преобразований изображений 
(аффинные и перспективные преобразования, изменение яркости, контраста), а также параметров алгоритмов на результаты 
обнаружения локальных особенностей
"""


# Corner detection

def cornerHarris(img, val):
    src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (5, 5))
    thresh = val
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.05

    # Detecting corners
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)

    # Normalizing
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    return img


def Shi_Tomasi(img, val):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, val, 0.01, 15)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 2, (255, 0, 0), -1)
    return img


src1 = cv.imread("house.jpg")
src2 = cv.imread("room.jpg")

max_thresh = 255
thresh = 220  # initial threshold

cv.imwrite("corners_orig1.jpg", cornerHarris(src1.copy(), thresh))
cv.imwrite("corners_orig2.jpg", cornerHarris(src2.copy(), thresh))

cv.imwrite("shi_orig1.jpg", Shi_Tomasi(src1.copy(), 100))
cv.imwrite("shi_orig2.jpg", Shi_Tomasi(src2.copy(), 60))

plt.show()

# Transformation Affine

srcTri = np.array([[0, 0], [src1.shape[1] - 1, 0], [0, src1.shape[0] - 1]]).astype(np.float32)
dstTri = np.array([[0, src1.shape[1] * 0.1], [src1.shape[1] * 0.85, src1.shape[0] * 0.25],
                   [src1.shape[1] * 0.15, src1.shape[0] * 0.7]]).astype(np.float32)
warp_mat = cv.getAffineTransform(srcTri, dstTri)
warp_dst = cv.warpAffine(src1, warp_mat, (src1.shape[1], src1.shape[0]))

srcTri2 = np.array([[0, 0], [src2.shape[1] - 1, 0], [0, src2.shape[0] - 1]]).astype(np.float32)
dstTri2 = np.array([[0, src2.shape[1] * 0.1], [src2.shape[1] * 0.85, src2.shape[0] * 0.25],
                    [src2.shape[1] * 0.15, src2.shape[0] * 0.7]]).astype(np.float32)
warp_mat2 = cv.getAffineTransform(srcTri2, dstTri2)
warp_dst2 = cv.warpAffine(src2, warp_mat2, (src2.shape[1], src2.shape[0]))
prespect_dst2 = cv.warpAffine(src1, warp_mat, (src1.shape[1], src1.shape[0]))

cv.imwrite("corners_wrap1.jpg", cornerHarris(warp_dst.copy(), thresh))
cv.imwrite("corners_wrap2.jpg", cornerHarris(warp_dst2.copy(), thresh))

cv.imwrite("shi_wrap1.jpg", Shi_Tomasi(warp_dst.copy(), 100))
cv.imwrite("shi_wrap2.jpg", Shi_Tomasi(warp_dst2.copy(), 60))

# Transformation Perspective

dst1 = np.array([
    [0, 0],
    [src1.shape[1], 0],
    [src1.shape[1] - 100, src1.shape[0] - 10],
    [60, src1.shape[0] - 100]], dtype="float32")

src_p1 = np.array([
    [0, 0],
    [src1.shape[1], 0],
    [src1.shape[1], src1.shape[0]],
    [0, src1.shape[0]]], dtype="float32")

dst2 = np.array([
    [0, 0],
    [src2.shape[1], 0],
    [src2.shape[1] - 100, src2.shape[0] - 10],
    [60, src2.shape[0] - 100]], dtype="float32")

src_p2 = np.array([
    [0, 0],
    [src2.shape[1], 0],
    [src2.shape[1], src2.shape[0]],
    [0, src2.shape[0]]], dtype="float32")

M1 = cv.getPerspectiveTransform(src_p1, dst1)
perspective1 = cv.warpPerspective(src1, M1, (src1.shape[1], src1.shape[0]))

M2 = cv.getPerspectiveTransform(src_p2, dst2)
perspective2 = cv.warpPerspective(src2, M2, (src2.shape[1], src2.shape[0]))

cv.imwrite("corners_perspect1.jpg", cornerHarris(perspective1.copy(), thresh))
cv.imwrite("corners_perspect2.jpg", cornerHarris(perspective2.copy(), thresh))

cv.imwrite("shi_perspect1.jpg", Shi_Tomasi(perspective1.copy(), 100))
cv.imwrite("shi_perspect2.jpg", Shi_Tomasi(perspective2.copy(), 60))
