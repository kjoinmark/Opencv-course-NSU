import cv2 as cv
import numpy as np

img = cv.imread('out0_0.jpg')

ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
# ycrcb = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
channels = cv.split(ycrcb)
edges = np.zeros(np.shape(channels[0]), np.uint8)

for channel in channels:
    ch = cv.blur(channel, (5, 5))
    canny = cv.Canny(ch, 100, 200, 5)
    edges += canny
    # cv.imshow("ch", ch)
    # cv.waitKey()

edges = cv.blur(edges, (6, 6))
cv.imshow("edges", edges)

edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
edges = cv.erode(edges, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))

# cv.imshow("edges", edges)
cv.waitKey()
contours1, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

res = cv.drawContours(edges, contours1, -1, 255, 3)
cv.imshow("res0", res)
# cv.waitKey()

contours, hierarchy = cv.findContours(res, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
res = cv.drawContours(res, contours, -1, 255, 3)

#
# cnt = contours[0]
# M = cv.moments(cnt)
# area = cv.contourArea(cnt)

# epsilon = 0.1 * cv.arcLength(cnt, True)
# approx = cv.approxPolyDP(cnt, epsilon, True)
# print(approx)
# area1 = cv.contourArea(approx)
# print(area, area1)
# res = img.copy()

max = 0
n_max = 0

cv.imshow("res1", res)
# cv.waitKey()


for i, contour in enumerate(contours):
    length = cv.arcLength(contour, True)
    contour = cv.approxPolyDP(contour, length * 0.1, True)
    hull = cv.convexHull(contour)
    length = cv.arcLength(hull, True)
    if (length > max):
        max = length
        n_max = i
        max_countor = hull.copy()
        print(hull, np.shape(hull))
    print(cv.isContourConvex(hull), cv.arcLength(hull, True))

print(max, n_max)
# res = cv.drawContours(res, contours, -1, (0, 255, 255), 1)

count_l = cv.arcLength(max_countor, True)
main_contour = max_countor

res_f = cv.drawContours(img, main_contour, -1, (0, 255, 0), 3)

rect = cv.minAreaRect(max_countor)
box = cv.boxPoints(rect)
box = np.int0(box)
# cv.drawContours(res_f,[box],-1,(0,0,255),2)

ellipse = cv.fitEllipse(max_countor)
cv.ellipse(res_f, ellipse, (0, 255, 0), 2)

# rect = cv.minAreaRect(main_contour)
# print(rect)
# ellipse = cv.fitEllipse(main_contour)
# print(ellipse)
# print(rect[1][1] * rect[1][0], ellipse[1][1] * ellipse[1][0] * np.pi / 4)

# contours, hierarchy = cv.findContours(channel, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
# res_contours = np.zeros(np.shape(contours), type(contours))


cv.imwrite('edges.jpg', edges)
cv.imwrite('result.jpg', res_f)

"""""
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
"""""
