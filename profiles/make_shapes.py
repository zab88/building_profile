import numpy as np
import cv2
import joblib


img_gray = cv2.imread('../imgs/p_all.jpg', 0)
th_box = int(img_gray.shape[0] * 0.007) * 2 + 1
th_box_div = 3
img_b = cv2.adaptiveThreshold(img_gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, th_box, th_box//th_box_div)

try:
    image_, contours_, hierarchy_ = cv2.findContours(img_b.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
except:
    contours_, hierarchy_ = cv2.findContours(img_b.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0
names = ['Barge Cap', 'Corri Apron', 'Deck Apron', 'Box Gutter', 'S Flashing', 'L Flashing']
contours_res = []
for cnt in contours_:
    x, y, w, h = cv2.boundingRect(cnt)

    if h < 40 or h > 400:
        continue

    i += 1
    if i == 6:
        cv2.rectangle(img_gray, (x, y), (x+w, y+h), (123,), 1)

    contours_res.append(cnt[:])

cv2.imshow('ss', img_gray)
cv2.waitKey()

res = {'names': names, 'cnt': contours_res}
joblib.dump(res, '../data/profile_names.pkl')

# https://stackoverflow.com/questions/55529371/opencv-shape-matching-between-two-similar-shapes
