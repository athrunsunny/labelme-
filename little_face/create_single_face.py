# -*- coding: utf-8 -*-
"""
Time:     2021.10.26
Author:   Athrunsunny
Version:  V 0.1
File:     yolotolabelme.py
Describe: Functions in this file is change the dataset format to labelme json file
"""
import base64
import io
import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm
import PIL.Image
import torch.nn.functional as F
import torch

ROOT_DIR = os.getcwd()
VERSION = '5.0.1'  # 根据labelme的版本来修改

FACE_KEYPOINT = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def process_point(points, cls):
    info = list()
    for idx, point in enumerate(points):
        shape_info = dict()
        if point is None:
            shape_info['label'] = ''
            shape_info['points'] = [[], []]
            shape_info['group_id'] = None
            shape_info['shape_type'] = 'rectangle'
            shape_info['flags'] = dict()
            info.append(shape_info)
            continue

        if cls[int(point[0])] == 'face':
            if point is None:
                shape_info['label'] = ''
                shape_info['points'] = [[], []]
            else:
                shape_info['label'] = cls[int(point[0])] + '_' + str(idx)
                shape_info['points'] = [[point[1], point[2]],
                                        [point[3], point[4]]]

            shape_info['group_id'] = None
            shape_info['shape_type'] = 'rectangle'
            shape_info['flags'] = dict()
            info.append(shape_info)

            for i in range(5):
                keypoint_info = dict()
                new_point = point[5:]
                keypoint_info['label'] = FACE_KEYPOINT[i] + '_' + str(idx)
                keypoint_info['points'] = [[new_point[i * 2], new_point[i * 2 + 1]]]
                keypoint_info['group_id'] = None
                keypoint_info['shape_type'] = 'point'
                keypoint_info['flags'] = dict()
                info.append(keypoint_info)

        else:
            # 仅有边界框的目标
            if point is None:
                shape_info['label'] = ''
                shape_info['points'] = [[], []]
            else:
                shape_info['label'] = cls[int(point[0])]
                shape_info['points'] = [[point[1], point[2]],
                                        [point[3], point[4]]]
            shape_info['group_id'] = None
            shape_info['shape_type'] = 'rectangle'
            shape_info['flags'] = dict()
            info.append(shape_info)
    return info


def create_json(img, imagePath, filename, info):
    data = dict()
    data['version'] = VERSION
    data['flags'] = dict()
    data['shapes'] = info
    data['imagePath'] = imagePath + '.jpg'
    height, width = img.shape[:2]
    # data['imageData'] = img_arr_to_b64(img).decode('utf-8')
    data['imageData'] = None
    data['imageHeight'] = height
    data['imageWidth'] = width
    jsondata = json.dumps(data, indent=4, separators=(',', ': '))
    f = open(filename, 'w')
    f.write(jsondata)
    f.close()


def read_txt(path):
    assert os.path.exists(path)
    with open(path, mode='r', encoding="utf-8") as f:
        content = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float64)
    res = np.unique(content, axis=0)
    if len(res) == 0:
        return None
    return res


def load_dataset_info(path=ROOT_DIR):
    yamlpath = glob(path + "\\*.yaml")[0]
    with open(yamlpath, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def reconvert_list(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] / dw
    w = box[2] / dw
    y = box[1] / dh
    h = box[3] / dh
    x1 = ((x + 1) * 2 - w) / 2.
    y1 = ((y + 1) * 2 - h) / 2.
    x2 = ((x + 1) * 2 + w) / 2.
    y2 = ((y + 1) * 2 + h) / 2.
    return x1, y1, x2, y2


def reconvert_np(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[:, :1] / dw
    w = box[:, 2:3] / dw
    y = box[:, 1:2] / dh
    h = box[:, 3:4] / dh
    box[:, :1] = ((x + 1) * 2 - w) / 2.
    box[:, 2:3] = ((x + 1) * 2 + w) / 2.
    box[:, 1:2] = ((y + 1) * 2 - h) / 2.
    box[:, 3:4] = ((y + 1) * 2 + h) / 2.
    return box


def reconvert_np_lmk(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    box[:, :1] = box[:, :1] / dw
    box[:, 1:2] = box[:, 1:2] / dh
    box[:, 2:3] = box[:, 2:3] / dw
    box[:, 3:4] = box[:, 3:4] / dh
    box[:, 4:5] = box[:, 4:5] / dw
    box[:, 5:6] = box[:, 5:6] / dh
    box[:, 6:7] = box[:, 6:7] / dw
    box[:, 7:8] = box[:, 7:8] / dh
    box[:, 8:9] = box[:, 8:9] / dw
    box[:, 9:] = box[:, 9:] / dh
    return box


def txt2json(proctype, cls, path=ROOT_DIR, padsz=5):
    process_image_path = os.path.join(path, proctype, 'images')
    process_label_path = os.path.join(path, proctype, 'labels')

    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    imgfiles = list()
    for extern in externs:
        imgfiles.extend(glob(process_image_path + "\\*." + extern))

    createfile = os.path.join(ROOT_DIR, 'createjson')
    if not os.path.exists(createfile):
        os.makedirs(createfile)

    for image_path in tqdm(imgfiles):
        # print(image_path)
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        size = (width, height)

        imgfilename = image_path.replace("\\", "/").split("/")[-1]
        imgname = '.'.join(imgfilename.split('.')[:-1])
        jsonpath = os.path.join(createfile, imgname + '.json')

        txtpath = os.path.join(process_label_path, imgname + '.txt')
        label_and_point = read_txt(txtpath)
        if label_and_point is not None:
            label_and_point[:, 1:5] = reconvert_np(size, label_and_point[:, 1:5])
            label_and_point[:, 5:] = reconvert_np_lmk(size, label_and_point[:, 5:])
            info = process_point(label_and_point, cls)
        else:
            info = process_point([label_and_point], cls)
        create_json(frame, imgname, jsonpath, info)
        shutil.copy(image_path, createfile)

        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # X = torch.tensor(image).transpose(0, 2).transpose(2, 1)
        # print("shape:", X.shape)
        # Ppadsz = padsz + 5
        # dim = (Ppadsz, Ppadsz, Ppadsz, Ppadsz)  # left,right,top ,down
        # X = F.pad(X, dim, "constant", value=114).transpose(2, 1).transpose(0, 2)
        # frame = X.data.numpy()

        def create_single_face(infos, image, name, path=ROOT_DIR, padsz=20):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            single_face_save_path = os.path.join(path, 'single_face')
            if not os.path.exists(single_face_save_path):
                os.makedirs(single_face_save_path)

            height, width = image.shape[:2]
            size = (width, height)

            for info in infos:
                if info['label'].split('_')[0] != 'face':
                    continue
                face_area_name = info['label']
                area = info['points']
                x1, y1 = area[0]
                x2, y2 = area[1]
                x1 -= padsz
                y1 -= padsz
                x2 += padsz
                y2 += padsz

                x1pad=y1pad=x2pad=y2pad =0
                if x1 < 0:
                    x1pad = int(abs(x1))
                if y1 < 0:
                    y1pad = int(abs(y1))
                if x2 > width - 1 :
                    x2pad = int(x2 - width)
                if y2 > height - 1 :
                    y2pad = int(y2 - height)

                x1 = x1 if x1 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                x2 = x2 if x2 < width - 1 else width - 1
                y2 = y2 if y2 < height - 1 else height - 1

                tmp_img = image[int(y1):int(y2), int(x1):int(x2), :]

                X = torch.tensor(tmp_img).transpose(0, 2).transpose(2, 1)
                dim = (x1pad, x2pad, y1pad, y2pad)  # left,right,top ,down
                X = F.pad(X, dim, "constant", value=114).transpose(2, 1).transpose(0, 2)
                tmp_img = X.data.numpy()

                def get_points_info(info,name,padsz=20):
                    point_info = []
                    idx = name.split('_')[-1]

                    def points_shift(info, x1, y1):
                        new_point = info
                        point = info['points'][0]
                        for i in range(len(point)):
                            if point[i] < 0:
                                continue
                            if i % 2 == 0:
                                point[i] = point[i] - x1
                            else:
                                point[i] = point[i] - y1

                        new_point['points'] = [point]
                        return new_point

                    for item in info:
                        if item['label'] == '_'.join([FACE_KEYPOINT[0],idx]):
                            point_info.append(points_shift(item,x1, y1))
                        if item['label'] == '_'.join([FACE_KEYPOINT[1],idx]):
                            point_info.append(points_shift(item,x1, y1))
                        if item['label'] == '_'.join([FACE_KEYPOINT[2],idx]):
                            point_info.append(points_shift(item,x1, y1))
                        if item['label'] == '_'.join([FACE_KEYPOINT[3],idx]):
                            point_info.append(points_shift(item,x1, y1))
                        if item['label'] == '_'.join([FACE_KEYPOINT[4],idx]):
                            point_info.append(points_shift(item,x1, y1))
                    return point_info
                point_info = get_points_info(infos,face_area_name,padsz=padsz)


                image_name = name + '.' + face_area_name + '.jpg'
                json_path_part = os.path.join(single_face_save_path, name + '.' + face_area_name + '.json')
                save_path = os.path.join(single_face_save_path, image_name)
                cv2.imwrite(save_path, tmp_img)
                create_json(tmp_img, name + '.' + face_area_name, json_path_part, point_info)

        create_single_face(info, frame, imgname,padsz=padsz)


def yolotolabelme(path=ROOT_DIR):
    pathtype = list()
    if 'train' in os.listdir(path):
        pathtype.append('train')
    if 'valid' in os.listdir(path):
        pathtype.append('valid')
    if 'test' in os.listdir(path):
        pathtype.append('test')

    cls = load_dataset_info()['names']

    for file_type in pathtype:
        print("Processing image type {} \n".format(file_type))
        txt2json(file_type, cls)


if __name__ == "__main__":
    yolotolabelme()
