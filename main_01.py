import cv2
import numpy as np
from helpers import get_8_parts, get_profile


img_file_in = 'imgs/ProfileReader1.jpg'
img_origin = cv2.imread(img_file_in)
img_gray = cv2.imread(img_file_in, 0)

img_draw = img_origin.copy()


th_box = int(img_gray.shape[0] * 0.007) * 2 + 1
th_box_div = 3
img_bin = cv2.adaptiveThreshold(img_gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, th_box, th_box//th_box_div)
# if kernel_size is not None:
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     img_bin = cv2.erode(img_bin, kernel)
# best_bin = is_id(img_bin.copy())

rec_8 = get_8_parts(img_bin)
for r in rec_8:
    x, y, w, h = r[0], r[1], r[2], r[3]

    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 1)

    img_dd = img_draw[y:y+h, x:x+w, :]  #.copy()
    best_c = get_profile(img_bin[y:y+h, x:x+w])
    if best_c is None:
        continue
    cv2.drawContours(img_dd, [best_c], -1, (0, 255, 0), 3)
    # cv2.imshow('asdfg', img_dd)
    # cv2.waitKey()


cv2.imshow('asdf', img_draw)
cv2.waitKey()

cv2.imwrite('out.jpg', img_draw)
