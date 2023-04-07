# -*- coding: utf-8 -*-
"""
Time:     2021.10.26
Author:   Athrunsunny
Version:  V 0.1
File:     yolotolabelme.py
Describe: Functions in this file is change the dataset format to labelme json file
"""
import base64
import copy
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

ROOT_DIR = os.getcwd()
VERSION = '5.0.1'  # 根据labelme的版本来修改

FACE_KEYPOINT = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']


def create_json(filename, info):
    jsondata = json.dumps(info, indent=4, separators=(',', ': '))
    f = open(filename, 'w')
    f.write(jsondata)
    f.close()


def resume(path=ROOT_DIR, padsz=20):
    resume_path = os.path.join(path, 'resume')
    if not os.path.exists(resume_path):
        os.makedirs(resume_path)

    createfile = os.path.join(path, 'createjson')
    jsonfiles = glob(createfile + '\\*.json')

    single_face = os.path.join(path, 'single_face')
    singlefacefiles = glob(single_face + '\\*.json')
    singlefacefiles_img = glob(single_face + '\\*.jpg')

    for file in tqdm(jsonfiles):
        imgfilename = file.replace("\\", "/").split("/")[-1]
        imgname = '.'.join(imgfilename.split('.')[:-1])
        image_path = file[:-4] + 'jpg'
        # frame = cv2.imread(image_path)

        new_json_path = os.path.join(resume_path, imgname + '.json')
        data = json.load(open(file, "r", encoding="utf-8"))
        new_data = copy.deepcopy(data)
        shapes = data['shapes']
        for item in shapes:
            if item['label'].split('_')[0] != 'face':
                continue
            face_area_name = item['label']
            single_face_json_name = imgname + '.' + face_area_name + '.json'
            single_face_json_img_name = imgname + '.' + face_area_name + '.jpg'
            single_face_json_path = os.path.join(single_face, single_face_json_name)
            single_face_json_img_path = os.path.join(single_face, single_face_json_img_name)
            if single_face_json_path in singlefacefiles:
                # 找到小图中的标注
                facedata = json.load(open(single_face_json_path, "r", encoding="utf-8"))
                faceshapes = facedata['shapes']

                lmk_points = [[] for _ in range(5)]
                for idx, faceitem in enumerate(faceshapes):
                    if faceitem['label'] == FACE_KEYPOINT[0] or faceitem['label'].split('_')[0] == FACE_KEYPOINT[0] or '_'.join(faceitem['label'].split('_')[:-1]) == FACE_KEYPOINT[0]:
                        lmk_points[0] = faceitem['points']
                    if faceitem['label'] == FACE_KEYPOINT[1] or faceitem['label'].split('_')[0] == FACE_KEYPOINT[1] or '_'.join(faceitem['label'].split('_')[:-1]) == FACE_KEYPOINT[1]:
                        lmk_points[1] = faceitem['points']
                    if faceitem['label'] == FACE_KEYPOINT[2] or faceitem['label'].split('_')[0] == FACE_KEYPOINT[2] or '_'.join(faceitem['label'].split('_')[:-1]) == FACE_KEYPOINT[2]:
                        lmk_points[2] = faceitem['points']
                    if faceitem['label'] == FACE_KEYPOINT[3] or faceitem['label'].split('_')[0] == FACE_KEYPOINT[3] or '_'.join(faceitem['label'].split('_')[:-1]) == FACE_KEYPOINT[3]:
                        lmk_points[3] = faceitem['points']
                    if faceitem['label'] == FACE_KEYPOINT[4] or faceitem['label'].split('_')[0] == FACE_KEYPOINT[4] or '_'.join(faceitem['label'].split('_')[:-1]) == FACE_KEYPOINT[4]:
                        lmk_points[4] = faceitem['points']
                lmk_points_np = np.array(lmk_points).reshape([1, 10])[0] - padsz

                top_left_x_shift = item['points'][0][0] - int(item['points'][0][0])
                top_left_y_shift = item['points'][0][1] - int(item['points'][0][1])

                lmk_points_np[0::2] += item['points'][0][0]
                lmk_points_np[1::2] += item['points'][0][1]

                lmk_points = lmk_points_np.tolist()

                keypoint_idx = item['label'].split('_')[-1]
                keypoint_info = list()
                keypoint_name = list()
                new_keypoint_info = list()
                for kpoints in data["shapes"]:
                    if ('_'.join(kpoints['label'].split('_')[:2]) in FACE_KEYPOINT or kpoints['label'].split('_')[
                        0] in FACE_KEYPOINT) and kpoints['label'].split('_')[-1] == keypoint_idx:

                        if len(kpoints['label'].split('_')) == 3:
                            k_idx = FACE_KEYPOINT.index('_'.join(kpoints['label'].split('_')[:2]))
                        else:
                            k_idx = FACE_KEYPOINT.index(kpoints['label'].split('_')[0])
                        new_kpoint = copy.deepcopy(kpoints)
                        new_kpoint['points'] = [lmk_points[2 * k_idx:2 * k_idx + 2]]
                        new_keypoint_info.append(new_kpoint)
                        keypoint_info.append(kpoints)
                        keypoint_name.append(kpoints['label'])
                if len(keypoint_info) != 0:
                    for new_item in new_data['shapes']:
                        if new_item['label'] in keypoint_name:
                            n_idx = new_data['shapes'].index(new_item)
                            kp_idx = keypoint_name.index(new_item['label'])
                            new_item = new_keypoint_info[kp_idx]
                            new_data['shapes'][n_idx] = new_item
            else:
                # 如果没有小人脸图的标注
                if single_face_json_img_path in singlefacefiles_img:
                    lmk_points = [-1 for _ in range(10)]
                    keypoint_idx = item['label'].split('_')[-1]
                    keypoint_info = list()
                    keypoint_name = list()
                    new_keypoint_info = list()
                    for kpoints in data["shapes"]:
                        if ('_'.join(kpoints['label'].split('_')[:2]) in FACE_KEYPOINT or kpoints['label'].split('_')[
                            0] in FACE_KEYPOINT) and kpoints['label'].split('_')[-1] == keypoint_idx:
                            if len(kpoints['label'].split('_')) == 3:
                                k_idx = FACE_KEYPOINT.index('_'.join(kpoints['label'].split('_')[:2]))
                            else:
                                k_idx = FACE_KEYPOINT.index(kpoints['label'].split('_')[0])
                            new_kpoint = copy.deepcopy(kpoints)
                            new_kpoint['points'] = [lmk_points[2 * k_idx:2 * k_idx + 2]]
                            new_keypoint_info.append(new_kpoint)
                            keypoint_info.append(kpoints)
                            keypoint_name.append(kpoints['label'])

                    # 如果有原始标注
                    if keypoint_info[0]['points'][0][0] <= -1 and keypoint_info[0]['points'][0][1] <= -1:
                        # 原始标注为-1 不做处理
                        pass
                    elif keypoint_info[0]['points'][0][0] != -1 and keypoint_info[0]['points'][0][1] != -1:
                        # 需要将标签中的人脸置零
                        for new_item in new_data['shapes']:
                            if new_item['label'] in keypoint_name:
                                n_idx = new_data['shapes'].index(new_item)
                                kp_idx = keypoint_name.index(new_item['label'])
                                new_item = new_keypoint_info[kp_idx]
                                new_data['shapes'][n_idx] = new_item


        create_json(new_json_path, new_data)
        shutil.copy(image_path, resume_path)


if __name__ == '__main__':
    path = r'D:\Users\yl3146\Desktop\tta\multask\justface'
    resume(path=path)
