import cv2
import numpy as np


def get_8_parts(img_bin: np.array) -> list:
    image_, contours_, hierarchy_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for cnt in contours_:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 0.4 * img_bin.shape[1] or w < 0.3 * img_bin.shape[1]:
            continue

        if h > 0.2 * img_bin.shape[0] or h < 0.12 * img_bin.shape[0]:
            continue

        res.append([x, y, w, h])

    return res


def get_profile(img_bin: np.array):
    img_b = img_bin.copy()

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # img_b = cv2.erode(img_b, kernel)

    image_, contours_, hierarchy_ = cv2.findContours(img_b.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # searching for biggerst contour
    biggest_c = -1
    best_c = None
    for cnt in contours_:
        x, y, w, h = cv2.boundingRect(cnt)

        # max_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, perimeter*0.05, True)

        if w > 0.9*img_bin.shape[1] or h > 0.9*img_bin.shape[0]:
            continue

        if w + h > biggest_c:
            biggest_c = w + h
            # best_c = cnt.copy()
            best_c = approx.copy()

    return best_c
