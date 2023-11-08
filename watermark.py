import math as m
import os
import random

import cv2
import numpy as np


class Watermark:
    def __init__(self,
                 wt_path) -> None:
        self.pokes = self.loadWatermark(wt_path)
        pass

    def location(self, w, h, dx, dy):
        y, x = random.randint(0, w-dy-1), random.randint(0, h-dx-1)
        return y, x

    def embedding(self, input):
        wt = random.choice(self.pokes)

        w, h, _ = input.shape

        img_area = w*h
        mask_area = random.randint(10, 20) * img_area / 100

        new_size = int(m.sqrt(mask_area))

        wt = cv2.resize(wt, (new_size, new_size))

        dx, dy, _ = wt.shape

        mask = cv2.cvtColor(wt, cv2.COLOR_BGR2GRAY)

        mask = np.where(mask > 240, 0, 255)

        cv2.imwrite("mask.png", mask)

        y, x = self.location(w, h, dx, dy)

        # print(w, h, x, y, dx, dy)

        for i in range(3):
            print(input.shape, y+dx, x+dy)
            tmp = input[y:y+dx, x:x+dy, i]
            new_img_np = np.where(mask == 0, tmp, wt[:, :, i])
            input[y:y+dx, x:x+dy, i] = new_img_np
        output = input

        out_mask = np.zeros(input.shape, dtype=np.uint8)
        out_mask = cv2.cvtColor(out_mask, cv2.COLOR_BGR2GRAY)
        out_mask[y:y+dx, x:x+dy] = mask

        print(out_mask.shape)
        return output, out_mask

    def loadImage(self):
        pass

    def loadWatermark(self, wt_path):
        wtms = os.listdir(wt_path)

        marks = []

        for item in wtms:
            marks.append(cv2.imread(os.path.join(wt_path, item)))

        print(len(marks))

        return marks


def gen_data(data_path, save_path, label_path, watermark):

    images = os.listdir(data_path)

    for img_name in images:
        print(img_name)
        img = cv2.imread(os.path.join(data_path, img_name))

        new_image, label = watermark.embedding(img)

        cv2.imwrite(os.path.join(save_path, img_name), new_image)
        cv2.imwrite(os.path.join(label_path, img_name), label)

    pass


if __name__ == "__main__":

    wt_path = "/home/xuan/BKU/Thesis/data/watermark"
    data_path = "/home/xuan/BKU/Thesis/data/sample"
    save_path = "/home/xuan/BKU/Thesis/data/result"
    label_path = "/home/xuan/BKU/Thesis/data/label"
    watermark = Watermark(wt_path)

    gen_data(data_path, save_path, label_path, watermark)
