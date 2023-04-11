# -*- coding: utf-8 -*-
"""
Time:     2021.10.26
Author:   Athrunsunny
Version:  V 0.1
File:     toyolo.py
Describe: Functions in this file is change the dataset format to yolov5
"""

import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT_DIR = os.getcwd()


def rename_files(file_name, path=ROOT_DIR):

    image_file_path = os.path.join(path,'images')
    label_file_path = os.path.join(path,'labels')

    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    image_files = list()
    for extern in externs:
        image_files.extend(glob(image_file_path + "\\*." + extern))

    txt_files = glob(label_file_path + "\\*.txt")

    assert len(image_files) == len(txt_files)

    des_path = os.path.join(path,'rename')
    des_img_path = os.path.join(des_path,'images')
    des_label_path = os.path.join(des_path, 'labels')
    if not os.path.exists(des_img_path):
        os.makedirs(des_img_path)
    if not os.path.exists(des_label_path):
        os.makedirs(des_label_path)

    for idx,(img,label) in tqdm(enumerate(zip(image_files,txt_files))):
        sif = str(idx)
        new_name = file_name + '_' +'0' * (4-len(sif)) + sif
        img_name = img.replace("\\", "/").split("/")[-1]
        label_name = label.replace("\\", "/").split("/")[-1]
        assert img_name.split('.')[0] == label_name.split('.')[0]

        new_img_name = new_name + '.' + img.split('.')[-1]
        new_label_name = new_name + '.' + label_name.split('.')[-1]
        new_img_name_path = os.path.join(des_img_path,new_img_name)
        new_label_name_path = os.path.join(des_label_path,new_label_name)

        os.rename(img,new_img_name_path)
        os.rename(label,new_label_name_path)

        # shutil.copy(new_img_name,des_img_path)
        # shutil.copy(new_label_name,des_label_path)

if __name__ == '__main__':
    file_path = r'G:\hp_tracking_proj\three_identy_data\65_sit\train'
    rename_files('65_sit',file_path)









