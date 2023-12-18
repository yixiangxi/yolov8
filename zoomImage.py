#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/12/8 11:55
# @Author : HuaWei/WenZhiWei
import cv2
import os

'''
因为yolo模型最大输入图片分辨率为640*640，故不需要大图片
因此进行了等比例缩放处理
'''

# 原始图片路径
images_root = "C:\\Users\\lenovo\\PycharmProjects\\targetDetection\\images100"
# 缩放图像存储路径
targer_images_root = "C:\\Users\\lenovo\\PycharmProjects\\targetDetection\\image_transformed"
# 最大长宽设置为1280，并等比例缩放
max_size = 1280

images_name = os.listdir(images_root)
images_path = [os.path.join(images_root, image_name) for image_name in images_name]
print(images_path[:10])


def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    # 判断图片的长宽比率

    if width > height:
        img_new = cv2.resize(image, (max_size, int(height / width * max_size)))
    else:
        img_new = cv2.resize(image, (int(width / height * max_size), max_size))
    return img_new


for i, image_path in enumerate(images_path):
    print(i)
    try:

        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        resize_img = img_resize(img)
        # cvt_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
        targer_image_path = os.path.join(targer_images_root, image_path.split('\\')[-1])
        cv2.imwrite(targer_image_path, resize_img)
    # break
    except Exception as e:
        print(image_path)
        print(e)
