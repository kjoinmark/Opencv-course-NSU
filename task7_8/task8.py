import cv2 as cv
import numpy as np

img = cv.imread("lena.png", 0)
cv.imwrite("lena_gray.png", img)
# ycrcb = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
[m, n] = img.shape
print("Введите уровень шума")

# N = int(input())
# print(type(N))
N = 50


def create_noise(img, N):
    noise = np.zeros((m, n), dtype=type(img[0][0]))
    for i in range(0, m):
        for j in range(0, n):
            tmp = img[i][j] + np.random.randint(-N / 2, N / 2)
            # print(tmp)
            """""
            if tmp > 255:
                tmp = 255
            if tmp < 0:
                tmp = 0
            """""
            noise[i][j] = tmp
    return noise


k = 10

result = np.zeros((m, n), dtype=int)

for i in range(k):
    tmp = create_noise(img, N)
    if i * 3 < k:
        cv.imshow(f"img{i}", tmp)
    result += tmp
cv.waitKey()
result = result / k

res = np.zeros((m, n), dtype=np.uint8)
res = result

# cv.imshow("orig", img)
# cv.imshow("noise", result)

diff = np.sum(img - res) / (m * n)
print(diff)

# cv.waitKey()
cv.imwrite("result.jpg", res)
