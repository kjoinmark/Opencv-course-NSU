import cv2
import numpy as np

img = cv2.imread("test2.png", 0)

T = 150
height, width = img.shape
print(height, width, height * width)


def brightness_counter(img, height, width):
    i = 0
    for j in range(height):
        for k in range(width):
            if img[j, k] >= T:
                i += 1
    return i


print(brightness_counter(img, height, width) / (height * width) * 100)
