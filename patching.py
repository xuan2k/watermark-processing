import os

import cv2
import numpy as np

img_folder = "/home/xuan/BKU/Thesis/data/result"
lbl_folder = "/home/xuan/BKU/Thesis/data/label"

save_foler = "/home/xuan/BKU/Thesis/data/patch"


imgs = os.listdir(img_folder)

for img_name in imgs:
    img = cv2.imread(os.path.join(img_folder, img_name))
    lbl = cv2.imread(os.path.join(lbl_folder, img_name))

    new_img = np.where(lbl < 250, img, lbl)

    cv2.imwrite(os.path.join(save_foler, img_name), new_img)
