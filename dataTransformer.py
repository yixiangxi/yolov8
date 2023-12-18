#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/12/8 11:55
# @Author : HuaWei/WenZhiWei


'''
# 坐标值xyxy->坐标值cxywh->百分比percent_cxywh

原始数据是xyxy左上右下，
转换后的yolo格式是cxywh的数值中心点cxy，wh宽高。
能够直接用于yolo系列的模型训练。
不过最终得按照此次竞赛要求输出结果
'''

import cv2
import numpy as np

import os

images_name_map = {}
# 原始图片路径
images_root = "C:\\Users\\lenovo\\PycharmProjects\\targetDetection\\images100"
# 原始标注路径
labels_root = "C:\\Users\\lenovo\\PycharmProjects\\targetDetection\\annotations"
# 标注转换后存储路径
targer_labels_root = "C:\\Users\\lenovo\\PycharmProjects\\targetDetection\\annotations_transformed"

# 生成classes.txt存储路径
classes_path = os.path.join(targer_labels_root, "classes.txt")

images_name = os.listdir(images_root)
for image_name in images_name:
    images_name_map[image_name.split('.')[0]] = image_name

labels_name = os.listdir(labels_root)
labels_path = [os.path.join(labels_root, label_name) for label_name in labels_name]


def coordinate_cxywh2percent(coordinate_cxywh, height, width):
    coordinate_cxywh[:, 0] = coordinate_cxywh[:, 0] / width
    coordinate_cxywh[:, 1] = coordinate_cxywh[:, 1] / height
    coordinate_cxywh[:, 2] = coordinate_cxywh[:, 2] / width
    coordinate_cxywh[:, 3] = coordinate_cxywh[:, 3] / height

    return coordinate_cxywh


def xyxy2cxywh(x):
    # Convert nx4 boxes from
    # [x1, y1, x2, y2] to [c,x, y, w, h]
    # where xy1=top-left, xy2=bottom-right**
    y = np.copy(x)
    y = y.astype(np.float64)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


cls2idx_map = {}


def cls2idx(cls):
    cls_lst = []
    for c in cls:
        if c in cls2idx_map.keys():
            cls_lst.append(cls2idx_map[c])
        else:
            cls2idx_map[c] = len(cls2idx_map.keys())
            cls_lst.append(cls2idx_map[c])

    return np.array(cls_lst)[:, np.newaxis]


for i, label_path in enumerate(labels_path[:100]):

    label_name = label_path.split("\\")[-1].split(".")[0]

    img = cv2.imread(os.path.join(images_root, images_name_map[label_name]))

    height, width = img.shape[:2]
    # print(height,width)
    # try:

    with open(label_path, 'r',encoding='utf-8') as txt_file:
        lines = txt_file.readlines()

    cls_xyxys = np.genfromtxt(label_path, delimiter=' ', dtype=None, encoding='utf-8')
    print(type(cls_xyxys))
    print(cls_xyxys.shape)
    if len(cls_xyxys.shape) == 0:
        print("元组")
        cls_xyxys = np.array([cls_xyxys])
    print(cls_xyxys)

    cls = [cls_xyxy[0] for cls_xyxy in cls_xyxys]
    cls = cls2idx(cls)
    xyxy = np.array([list(cls_xyxy)[1:] for cls_xyxy in cls_xyxys])

    # print(cls,xyxy)
    cxywh = xyxy2cxywh(xyxy)
    percent_cxywh = coordinate_cxywh2percent(cxywh, height, width)

    target_np = np.concatenate((cls, percent_cxywh), axis=1)
    print(target_np)
    targer_label_path = os.path.join(targer_labels_root, label_path.split('\\')[-1])
    print(targer_label_path)
    np.savetxt(targer_label_path, target_np, fmt="%d %.6f %.6f %.6f %.6f", )

with open(classes_path, 'w') as classes_file:
    for cls in cls2idx_map.keys():
        classes_file.write(cls + '\n')

    # except Exception as e:
    # print(label_path)
    # print(e)
