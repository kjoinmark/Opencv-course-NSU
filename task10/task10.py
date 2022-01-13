import cv2 as cv
import numpy as np

img_rgb = cv.imread('image.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('template.jpg', 0)
w, h = template.shape[::-1]
rows, cols = np.shape(img_gray)


def task(gray):
    res = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where(res >= threshold)
    if np.size(loc) == 0:
        print("no match")
    else:
        print("it works")
    for pt in zip(*loc[::-1]):
        cv.rectangle(gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return gray


# поворот
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


rotate_img = rotateImage(img_gray.copy(), 6)
cv.imwrite("rotated.jpg", task(rotate_img))

# шум
img_noise = np.zeros((rows, cols), dtype=np.uint8)
cv.randu(img_noise, 0, 11)
img_noise = img_noise + img_gray
cv.imwrite("noise.jpg", task(img_noise))


# яркость

def increase_brightness(img, value=30):
    lim = 255 - value
    img[img > lim] = 255
    img[img <= lim] += value

    return img


img_bright = increase_brightness(img_gray.copy(), value=180)
cv.imwrite("brightness.jpg", task(img_bright))

# контраст

alpha = 4.1
beta = -40
img_contrast = cv.convertScaleAbs(img_gray.copy(), alpha=alpha, beta=beta)
cv.imwrite("contrast.jpg", task(img_contrast))

# масштаб

img_size = cv.resize(img_gray.copy(), None, fx=1.1, fy=1.1, interpolation=cv.INTER_CUBIC)
# img_size = cv.imread('image_res.jpg', 0)
cv.imwrite("size.jpg", task(img_size))

cv.imwrite("res.jpg", img_rgb)
cv.waitKey()
