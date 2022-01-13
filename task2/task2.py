import cv2
import numpy as np

img = cv2.imread("image.jpg")

T = 150

cv2.circle(img, (288, 303), 25, (100, 100, 0), 5)
cv2.ellipse(img, ((424, 319), (70, 40), 0), (100, 100, 0), 5)
ear1 = np.array([(538, 210), (695, 142), (610, 307)])
ear2 = np.array([(300, 186), (191, 119), (247, 239)])
food = np.array([(177, 650),
                 (176, 590),
                 (224, 567),
                 (298, 583),
                 (330, 629),
                 (274, 663),
                 (213, 655),
                 ])

cv2.polylines(img, [ear1], isClosed=True, color=(200, 100, 0), thickness=4)
cv2.polylines(img, [ear2], isClosed=True, color=(200, 100, 0), thickness=4)
cv2.polylines(img, [food], isClosed=True, color=(200, 100, 0), thickness=4)
cv2.imshow("img", img)

cv2.imwrite("result.jpg", img)
