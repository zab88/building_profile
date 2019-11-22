import cv2
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
from itertools import permutations


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

    res.sort(key=lambda x: x[0]+2*x[1])

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

        approx = cv2.approxPolyDP(cnt, perimeter*0.003, True)

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


def cnt2res(cnt):
    cc = cnt.copy()
    print('asdfss')

    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, perimeter * 0.15, True)
    print('sss')


def get_key_points(best_c, img_bin):
    hull_points = cv2.convexHull(best_c)
    hp = hull_points.copy()

    # need to delete too close points
    for i in range(5):
        close_points = []
        for k, el in enumerate(hp):
            k_1 = (k + 1) % hp.shape[0]
            d01 = np.linalg.norm(hp[k] - hp[k_1])
            if d01 < 6:
                close_points.append(k)
                # hp[k_1] = (hp[k]+hp[k_1])//2
                hp[k_1] = (hp[k]+hp[k])//2
        hp = np.delete(hp, close_points, 0)
    hull_points = hp.copy()
    #return hp

    # points_to_del = []
    # for k, el in np.ndenumerate(hull_points[0]):

    def iterative_deletion(hull_points):
        for k, el in enumerate(hull_points):
            if k+2 >= hull_points.shape[0]:
                continue
            k_1 = (k+1) % hull_points.shape[0]
            k_2 = (k+2) % hull_points.shape[0]

            d01 = np.linalg.norm(hull_points[k] - hull_points[k_1])
            d02 = np.linalg.norm(hull_points[k] - hull_points[k_2])
            d12 = np.linalg.norm(hull_points[k_1] - hull_points[k_2])

            # print(k, d01, d02, d12)
            if d01 > 5 and d12 > 5 and (d02*1.15 > d01+d12):
                return k_1
                #points_to_del.append(k_1)
        return None

    i_d = iterative_deletion(hull_points)
    while i_d is not None:
        hull_points = np.delete(hull_points, [i_d], 0)
        i_d = iterative_deletion(hull_points)
    hp = hull_points.copy()

    # return hp

    # making right order
    tmp_bin = np.zeros(img_bin.shape, np.uint8)
    tmp_bin[:, :] = 255
    cv2.drawContours(tmp_bin, [best_c], -1, (0,), -1)
    # make all possible variants
    all_combo = permutations(hp, hp.shape[0])
    best_combo, max_zero = None, -1
    for combo in all_combo:
        tmp_bin_c = tmp_bin.copy()
        not_allowed = False
        zero_before = cv2.countNonZero(tmp_bin_c)
        # print(combo)
        for k, el in enumerate(combo):
            if k + 1 >= len(combo):
                continue
            # check posibility
            td_ = 8
            tmp_bin2 = tmp_bin.copy()
            patch = tmp_bin2[
                (combo[k][0][1]+combo[k+1][0][1])//2 - td_:(combo[k][0][1] + combo[k+1][0][1])//2 + td_,
                (combo[k][0][0]+combo[k+1][0][0])//2-td_:(combo[k][0][0]+combo[k+1][0][0])//2+td_
            ]
            # patch[:, :] = 80
            # cv2.imshow('bxc', tmp_bin2)
            # cv2.waitKey()
            if cv2.countNonZero(patch) >= patch.shape[0] * patch.shape[1]:
                not_allowed = True
                break

            # k_1 = (k + 1) % hp.shape[0]
            k_1 = (k + 1) % len(combo)
            # cv2.line(tmp_bin, (hp[k][0][1], hp[k][0][0]), (hp[k_1][0][1], hp[k_1][0][0]), (155,), 3)
            # cv2.line(tmp_bin_c, (hp[k][0][0], hp[k][0][1]), (hp[k_1][0][0], hp[k_1][0][1]), (155,), 3)
            cv2.line(tmp_bin_c, (combo[k][0][0], combo[k][0][1]), (combo[k_1][0][0], combo[k_1][0][1]), (155,), 5)
        zero_after = cv2.countNonZero(tmp_bin_c)
        if not_allowed is False and zero_after - zero_before > max_zero:
            max_zero = zero_after - zero_before
            best_combo = combo[:]
        # cv2.imshow('fsfsd', tmp_bin_c)
        # cv2.waitKey()

    if False:
        p_order = []
        # np_order = np.array([])
        np_order = []
        for k_c, el_c in enumerate(best_c):
            best_dd = 100000
            best_k = None
            for k_a, el_a in enumerate(hp):
                # if not(el_c[0][0] == el_a[0][0] and el_c[0][1] == el_a[0][1]):
                #     continue
                dd = np.linalg.norm(hp[k_a] - best_c[k_c])
                if dd < 10 and best_dd > dd:
                    best_dd = dd
                    best_k = k_a

            # if best_k is not None and best_k not in p_order:
            if best_k is not None:
                p_order.append(best_k)
                # np_order = np.append(np_order, hull_points[k_a], axis=0)
                np_order.append(hp[best_k])
        np_order = np.array(np_order, dtype=np.int)
        print('P_ORDER', p_order)

    # for i in range(hp.shape[0]):

    # print(hull_points[k], 'gg')
    # return hp
    return np.array(best_combo)
    # return np_order


def get_angles(points):
    angles = []
    for k, el in enumerate(points):
        k_1 = (k + 1) % points.shape[0]
        k_2 = (k + 2) % points.shape[0]
        if k + 2 >= points.shape[0]:
            continue

        u = - points[k] + points[k_1]
        v = points[k_1] - points[k_2]
        c = np.dot(u[0], v[0]) / np.linalg.norm(u) / np.linalg.norm(v)
        angle = np.arccos(np.clip(c, -1, 1))
        angles.append(np.degrees(angle))

    return angles


def get_digit_groups(digits, profile):
    # search 3 most close digits
    d_groups = []
    for k, p in enumerate(profile):
        if k+1 >= len(profile):
            continue
        d_group = []
        y_p, x_p = (p[0][1] + profile[k+1][0][1])//2, (p[0][0] + profile[k+1][0][0])//2
        # digits.sort(key=lambda d: np.linalg.norm( [x_p, y_p],  )
        for d in digits:
            if len(d_group) >= 3:
                continue
            #if (d[0]+d[2]/2 - x_p)*(d[0]+d[2]/2 - x_p) + (d[1]+d[3]/2 - y_p)*(d[1]+d[3]/2 - y_p) < 2300:
            if pow(d[0]+d[2]/2 - x_p, 2) + pow(d[1]+d[3]/2 - y_p, 2) < 2300:
            # if (d[1] - x_p)*(d[1] - x_p) + (d[0] - y_p)*(d[0] - y_p) < 50:
                if len(d_group) < 1:
                    d_group.append(d[:])
                elif len(d_group) == 1:
                    # if abs(d_group[0][0] - d[0]) < d[2] + d_group[0][2]:
                    #     d_group.append(d[:])
                    if abs(d_group[0][1] - d[1]) < (d[3] + d_group[0][3])*0.7:
                        d_group.append(d[:])
                    # d_group.append(d[:])
                else:
                    # check
                    # if min(abs(d_group[0][0] - d[0]), abs(d_group[1][0] - d[0])) < d[2] + d_group[0][2]:
                    #     d_group.append(d[:])
                    if min(abs(d_group[0][1] - d[1]), abs(d_group[1][1] - d[1])) < (d[3] + d_group[0][3])*0.6:
                        d_group.append(d[:])
                    # d_group.append(d[:])
                    # continue
        d_groups.append(d_group[:])
    return d_groups


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
