import cv2 as cv

"""
Протестировать алгоритмы OpenCV на эффективность удаление фона из видеопоследовательностей. Объяснить наблюдаемые эффекты
"""
backSub = cv.createBackgroundSubtractorMOG2()
backSub2 = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture('videoplayback.mp4')

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask1 = backSub.apply(frame)
    fgMask2 = backSub2.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('MOG2 Mask', fgMask1)
    cv.imshow('KNN Mask', fgMask2)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
