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
import copy

ROOT_DIR = os.getcwd()
VERSION = '5.0.1'  # 根据labelme的版本来修改

FACE_KEYPOINT = ['left_eye','right_eye','nose','left_mouth','right_mouth']


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


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
    for idx,point in enumerate(points):
        shape_info = dict()
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
            if point.shape[0] < 15:
                continue
            for i in range(5):
                keypoint_info = dict()
                new_point = point[5:]
                keypoint_info['label'] = FACE_KEYPOINT[i] + '_' + str(idx)
                keypoint_info['points'] = [[new_point[i*2], new_point[i*2 + 1]]]
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


def get_idx(labels,thresh):
    idx_list = list()
    for idx,item in enumerate(labels):
        if item[3] - item[1] < thresh and item[4] - item[2] < thresh:
            continue
        idx_list.append(idx)
    return idx_list


def txt2json(proctype, cls, path=ROOT_DIR,img_size=800):
    process_image_path = os.path.join(path, proctype, 'images')
    process_label_path = os.path.join(path, proctype, 'labels')

    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    imgfiles = list()
    for extern in externs:
        imgfiles.extend(glob(process_image_path + "\\*." + extern))

    createfile = os.path.join(ROOT_DIR, 'createjson')
    if not os.path.exists(createfile):
        os.makedirs(createfile)

    total_bbox = 0
    ori_total_bbox = 0
    for image_path in tqdm(imgfiles):
        # print(image_path)
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        ori_size = (width, height)

        h,w = height,width
        h0, w0 = frame.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            new_frame = cv2.resize(frame, (int(w0 * r), int(h0 * r)), interpolation=interp)
        else:
            new_frame = frame
        height, width = new_frame.shape[:2]
        size = (width, height)
        # im_name = im.replace("\\", "/").split("/")[-1]
        # image = cv2.imread(im)
        # image, ratio, pad = letterbox(frame, (img_size,img_size), auto=False)
        # shapes = (h0, w0), ((h / h0, w / w0), pad)

        # saveimage_path = os.path.join(des_path, im_name)


        imgfilename = image_path.replace("\\", "/").split("/")[-1]
        imgname = '.'.join(imgfilename.split('.')[:-1])
        jsonpath = os.path.join(createfile, imgname + '.json')

        txtpath = os.path.join(process_label_path, imgname + '.txt')
        label_and_point = read_txt(txtpath)
        ori_labels = copy.deepcopy(label_and_point)
        a = len(ori_labels)
        ori_total_bbox += a
        if label_and_point is not None:
            label_and_point[:, 1:5] = reconvert_np(size, label_and_point[:, 1:5])
            idx = get_idx(label_and_point,thresh=10)
            new_labels = ori_labels[idx]
            b = len(new_labels)
            # print(a==b)
            # label_and_point[:, 5:15] = reconvert_np_lmk(size, label_and_point[:, 5:15])
            # info = process_point(label_and_point, cls)
            new_labels[:, 1:5] = reconvert_np(ori_size, new_labels[:, 1:5])
            new_labels[:, 5:15] = reconvert_np_lmk(ori_size, new_labels[:, 5:15])
            info = process_point(new_labels[:, :5], cls)
            # if b != len(info):
                # print('1111111111')
            total_bbox += len(info)
        else:
            info = process_point([label_and_point], cls)
            total_bbox += len(info)

        create_json(frame, imgname, jsonpath, info)
        shutil.copy(image_path, createfile)
    print(total_bbox)
    print(ori_total_bbox)


def yolotolabelme(path=ROOT_DIR):
    pathtype = list()
    if 'train' in os.listdir(path):
        pathtype.append('train')
    if 'valid' in os.listdir(path):
        pathtype.append('valid')
    if 'test' in os.listdir(path):
        pathtype.append('test')

    cls = load_dataset_info()['names']
    cls = [cls[-1]]
    for file_type in pathtype:
        print("Processing image type {} \n".format(file_type))
        txt2json(file_type, cls)


if __name__ == "__main__":
    # image_size / input_size * stride
    # fliter condition 
    yolotolabelme()
