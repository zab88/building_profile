import cv2
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib


def get_8_parts(img_bin: np.array) -> list:
    # image_, contours_, hierarchy_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_, hierarchy_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for cnt in contours_:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 0.4 * img_bin.shape[1] or w < 0.3 * img_bin.shape[1]:
            continue

        if h > 0.22 * img_bin.shape[0] or h < 0.12 * img_bin.shape[0]:
            continue

        res.append([x, y, w, h])

    return res


def get_best_approx(cnt):
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    area_list = []
    for mult in [0.01, 0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(cnt, perimeter * mult, True)
        area_list.append(cv2.contourArea(np.array(approx)))
    print('original area:', area)
    print('other area:', area_list)


def get_profile(img_bin: np.array):
    img_b = img_bin.copy()
    # img_b = img_bin

    hh, ww = img_bin.shape[:2]
    # check that cell contains profile
    has_something = img_bin[int(hh*0.1):int(hh*0.9), int(ww*0.1):int(ww*0.9)]
    if np.count_nonzero(255 - has_something) < hh+ww:
        return None


    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # img_b = cv2.erode(img_b, kernel)

    # image_, contours_, hierarchy_ = cv2.findContours(img_b.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_, hierarchy_ = cv2.findContours(img_b.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # searching for biggerst contour
    biggest_c = -1
    best_c, best_cnt = None, None
    best_c2 = None
    for cnt in contours_:
        x, y, w, h = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > perimeter*3:
            continue

        approx = cv2.approxPolyDP(cnt, perimeter*0.01, True)

        if w > 0.78*img_bin.shape[1] or h > 0.9*img_bin.shape[0]:
            continue
        if w < 0.2 * img_bin.shape[1] and h < 0.2 * img_bin.shape[0]:
            continue

        if w + h > biggest_c:
            # if w > 0.4*img_bin.shape[1] or h > 0.4*img_bin.shape[0]:
            best_c2 = best_c.copy() if best_c is not None else None
            biggest_c = w + h
            # best_c = cnt.copy()
            best_c = approx.copy()
            best_cnt = cnt.copy()

    # get_best_approx(best_cnt)
    if True:
        tmp_bin = np.zeros(img_bin.shape, np.uint8)
        tmp_bin[:, :] = 255

        cv2.drawContours(tmp_bin, [best_c], -1, (0,), -1)
        if best_c2 is not None:
            cv2.drawContours(tmp_bin, [best_c2], -1, (0,), -1)

        kernel = np.ones((3, 3), np.uint8)
        tmp_bin = cv2.erode(tmp_bin, kernel)

        # cv2.imshow('fff', tmp_bin)
        # cv2.waitKey()

        # if merging correct, only one contour should be
        # image_, contours_, hierarchy_ = cv2.findContours(tmp_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_, hierarchy_ = cv2.findContours(tmp_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # print(len(contours_))
        if len(contours_) == 2:
            return contours_[1]

        # cv2.imshow('fff', tmp_bin)
        # cv2.waitKey()


    return best_c


# clf = joblib.load("data/digits_cls_lgbm.pkl")
clf = joblib.load("data/digits_cls.pkl")
def get_digit(img_bin: np.array):
    img_b = img_bin.copy()
    # img_b = img_bin

    hh, ww = img_bin.shape[:2]
    # check that cell contains profile
    has_something = img_bin[int(hh*0.1):int(hh*0.9), int(ww*0.1):int(ww*0.9)]
    if np.count_nonzero(255 - has_something) < hh+ww:
        return None

    b_w = 5
    # img_b = cv2.copyMakeBorder(img_b, b_w, b_w, b_w, b_w, cv2.BORDER_CONSTANT, value=(255,))
    # cv2.imshow('bwbw', img_b)
    # cv2.waitKey()
    # image_, contours_, hierarchy_ = cv2.findContours(img_b.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_, hierarchy_ = cv2.findContours(img_b.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    found_digits = []
    for cnt in contours_:
        x, y, w, h = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > perimeter * 5:
            continue

        approx = cv2.approxPolyDP(cnt, perimeter * 0.01, True)

        if w > 0.2 * img_bin.shape[1] or h > 0.2 * img_bin.shape[0]:
            continue
        if w < 3 or h < 7:
            continue
        if w < 0.05 * img_bin.shape[1] and h < 0.05 * img_bin.shape[0]:
            continue

        is_found_near = False
        for cnt2 in contours_:
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            if x == x2 and y == y2:
                continue
            if w2 > 0.2 * img_bin.shape[1] or h2 > 0.2 * img_bin.shape[0]:
                continue
            if w2 < 3 or h2 < 7:
                continue
            if w2 < 0.05 * img_bin.shape[1] and h2 < 0.05 * img_bin.shape[0]:
                continue

            area2 = cv2.contourArea(cnt2)
            perimeter2 = cv2.arcLength(cnt2, True)
            if area2 > perimeter2 * 3:
                continue

            if (x-x2)*(x-x2) + (y-y2)*(y-y2) < h*h*1.5:
                is_found_near = True
                break
        if not is_found_near:
            continue

        # Make the rectangular region around the digit
        img_bin_black = img_bin.copy()
        img_bin_black[0:y, :] = 255
        img_bin_black[y+h:, :] = 255
        img_bin_black[:, 0:x] = 255
        img_bin_black[:, x+w:] = 255
        # print(x, y, w, h, img_bin.shape)

        # cv2.imshow('fff', img_bin_black)
        # cv2.waitKey()

        b_part = max(w//2, h//2) + 1
        pt1 = max(0, int(x+w//2 - b_part))
        pt2 = max(0, int(y+h//2 - b_part))
        roi = img_bin_black[pt2:pt2 + 2*b_part, pt1:pt1 + 2*b_part]
        # cv2.imshow('fff2', roi)
        # cv2.waitKey()
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # cv2.imshow('fff2', roi)
        # cv2.waitKey()

        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))

        found_digits.append([x, y, w, h, nbr[0]])

    return found_digits
