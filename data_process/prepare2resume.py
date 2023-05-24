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


def create_info(name):
    point_info = []
    idx = name.split('_')[-1]
    for i in range(5):
        new_dict = {}
        new_dict['label'] = '_'.join([FACE_KEYPOINT[i], idx])
        new_dict['points'] = [[-1, -1]]
        new_dict['group_id'] = None
        new_dict['shape_type'] = 'point'
        new_dict['flags'] = {}
        point_info.append(new_dict)
    return point_info


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


def create_new_json(file_path):
    # 验证数据并生成json用
    data_jpg = glob(file_path + '\\*.jpg')
    data_json = glob(file_path + '\\*.json')
    for jpg in tqdm(data_jpg):
        img = cv2.imread(jpg)
        jpg_name = jpg.replace("\\", "/").split("/")[-1].split(".jpg")[0]
        json_path = os.path.join(file_path, jpg_name + '.json')
        json_name = jpg_name + '.json'
        face_name = jpg_name.split('.')[-1]
        info = create_info(face_name)
        if json_path in data_json:
            # 处理有文件但是为空的情况
            json_file = json.load(open(json_path, "r", encoding="utf-8"))
            if len(json_file['shapes']) == 0:
                json_file['shapes'] = info

                jsondata = json.dumps(json_file, indent=4, separators=(',', ': '))
                f = open(json_path, 'w')
                f.write(jsondata)
                f.close()
                print('1', json_name)
        else:
            # 处理没有json文件的情况
            create_json(img, jpg_name, json_path, info)
            print('2', json_name)


def prepare2resume_data(new_data_path, des_path, whole_data_path):
    # des_path = os.path.join(des_path, 'processed')
    # if not os.path.exists(des_path):
    #     os.makedirs(des_path)
    # 手动修改一下数据路径即可
    split_data_path = new_data_path
    clean_data_jpg = glob(split_data_path + '\\*.jpg')
    clean_data_json = glob(split_data_path + '\\*.json')

    whole_data_jpg = glob(whole_data_path + '\\*.jpg')

    clean_data_name_jpg = [i.replace("\\", "/").split("/")[-1].split(".jpg")[0] for i in clean_data_jpg]
    clean_data_name_json = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in clean_data_json]

    count = 0
    for data in tqdm(whole_data_jpg):
        data_name = data.replace("\\", "/").split("/")[-1].split(".jpg")[0]
        if data_name in clean_data_name_jpg:
            count += 1
            continue
        json_path = os.path.join(whole_data_path, data_name + '.json')
        shutil.copy(data, des_path)
        shutil.copy(json_path, des_path)

    print(count)


def preparedata(path=ROOT_DIR):
    pass


def copy_files(ori, des):
    if not os.path.exists(des):
        os.makedirs(des)

    files = glob(ori + '\\*')

    for file in tqdm(files):
        shutil.copy(file, des)



def move_files(ori, des):
    if not os.path.exists(des):
        os.makedirs(des)

    files = glob(ori + '\\*')

    for file in tqdm(files):
        shutil.move(file, des)


def copy_files1(new_data_path, des_path, whole_data_path):
    # des_path = os.path.join(des_path, 'processed')
    # if not os.path.exists(des_path):
    #     os.makedirs(des_path)
    # 手动修改一下数据路径即可
    split_data_path = new_data_path
    clean_data_txt = glob(split_data_path + '\\*.txt')
    # clean_data_json = glob(split_data_path + '\\*.json')

    whole_data_txt = glob(whole_data_path + '\\*.txt')

    clean_data_name_txt = [i.replace("\\", "/").split("/")[-1].split(".txt")[0] for i in clean_data_txt]
    # clean_data_name_json = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in clean_data_json]

    count = 0
    for data in tqdm(whole_data_txt):
        data_name = data.replace("\\", "/").split("/")[-1].split(".txt")[0]
        if data_name in clean_data_name_txt:
            count += 1
            continue
        # json_path = os.path.join(whole_data_path, data_name + '.json')
        shutil.copy(data, des_path)
        # shutil.copy(json_path, des_path)

    print(count)


def copy_files2(ori, des):
    if not os.path.exists(des):
        os.makedirs(des)

    files = glob(ori + '\\*')
    des_files = glob(des + '\\*')
    for file in tqdm(files):
        if file in des_files:
            continue
        shutil.copy(file, des)

if __name__ == "__main__":
    # prepare2resume_data()
    ori = r'G:\hp_tracking_proj\three_identy_data\wface_yolotype________val\createjson'
    des = r'G:\hp_tracking_proj\新标注规则标注的人脸框_验证集'
    #
    # copy_files(ori, des)
    move_files(ori, des)
    # print('create json files')
    # create_new_json(des)

    # new_data_path = r'G:\hp_tracking_proj\single_face_process\widerface_head_body_pose_lmk\wait_resume\all_clean\all_szclean'
    # whole_data_path = r'G:\hp_tracking_proj\single_face_process\widerface_head_body_pose_lmk\wait_resume\sz_clean\des_path'
    # des_path = r'G:\hp_tracking_proj\single_face_process\widerface_head_body_pose_lmk\wait_resume\all_clean\des_path_szclean'
    # prepare2resume_data(new_data_path,des_path,whole_data_path)
    # copy_files1(new_data_path,des_path,whole_data_path)
