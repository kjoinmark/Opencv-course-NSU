import cv2 as cv
import numpy as np

img_rgb = cv.imread('test.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

rows, cols = np.shape(img_gray)


def dota_time(image):
    win = cv.imread(f'win.jpg', 0)
    templates = []
    for i in range(0, 11):
        templates.append(cv.imread(f'templates\{i}.png', 0))

    res = cv.matchTemplate(image, win, cv.TM_CCOEFF_NORMED)

    threshold = 0.9
    loc = np.where(res >= threshold)

    cropped = image[int(loc[0] - 2):int(loc[0] + 14), int(loc[1] - 10):int(loc[1] + 120)]

    positions = []

    for i in range(11):
        res = cv.matchTemplate(cropped, templates[i], cv.TM_CCOEFF_NORMED)
        threshold = 0.85
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            positions.append([i, pt])

    positions.sort(key=lambda positions: positions[1])
    time = ""
    for i in positions:
        if i[0] == 10:
            time += ':'
        else:
            time += str(i[0])
    return time


print(dota_time(img_gray))
