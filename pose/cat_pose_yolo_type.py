import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm


def read_txt(path):
    assert os.path.exists(path)
    with open(path, mode='r', encoding="utf-8") as f:
        content = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float64)
    res = np.unique(content, axis=0)
    if len(res) == 0:
        return None
    return res


def cat(timage, tlabel, image_dir, label_dir, des_dir):
    """
    Args:
        timage: 使用6Dpose预测的图片
        tlabel: 使用6Dpose预测的标签
        image_dir: 原始图像路径
        label_dir: 原始标签路径
        des_dir: 最终保存路径

    Returns:

    """
    os.makedirs(des_dir, exist_ok=True)
    des_img_dir = os.path.join(des_dir, 'images')
    des_lab_dir = os.path.join(des_dir, 'labels')
    os.makedirs(des_img_dir, exist_ok=True)
    os.makedirs(des_lab_dir, exist_ok=True)

    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(timage + "\\*." + extern))

    for img in tqdm(files):
        image_name = '.'.join(img.replace('\\', '/').split("/")[-1].split('.')[:-1])
        ori_image_name = image_name.split('.')[0]
        indice = int(image_name.split('.')[-1].split('_')[-1])
        label_name = image_name + '.txt'
        label_path = os.path.join(tlabel, label_name)
        label_and_point = read_txt(label_path)[0]
        def change2reg(label):
            # pitch,yaw,roll = label[:,:1],label[:,1:2],label[:,2:3]
            label[:1] = label[:1] * np.pi / 180
            label[1:2] = -(label[1:2] * np.pi / 180)
            label[2:3] = label[2:3] * np.pi / 180
            return label

        label_and_point = change2reg(label_and_point)

        ori_image_path = os.path.join(image_dir, ori_image_name + '.jpg')
        ori_label_path = os.path.join(label_dir, ori_image_name + '.txt')
        ori_label_and_point = read_txt(ori_label_path)
        # ori_label_and_point[:,[5,-1]] = ori_label_and_point[:,[-1,5]]
        out_file = open('%s/%s.txt' % (des_lab_dir, ori_image_name), 'w')

        # out_file = open(ori_label_path, 'w')
        a = ori_label_and_point[ori_label_and_point[:, 5] == indice]
        row_index = np.where((ori_label_and_point == a).all(axis=1))[0]
        ori_label_and_point[row_index[0]][-3:] = label_and_point
        for item in ori_label_and_point:
            out_file.write(" ".join([str(a) for a in item[:]]) + '\n')


timage = r'G:\deeplearning_dataset\widerface_hpfl_yolo_new\pose_result_3k'
tlabel = r'G:\deeplearning_dataset\widerface_hpfl_yolo_new\pose_result_txt_3k'
image_dir = r'G:\deeplearning_dataset\widerface_hpfl_yolo_new\createjson_part\train\images'
label_dir = r'G:\deeplearning_dataset\widerface_hpfl_yolo_new\createjson_part\train\labels'
des_dir = r'G:\deeplearning_dataset\widerface_hpfl_yolo_new\createjson_part_with_pose\train'

# cat(timage, tlabel, image_dir, label_dir, des_dir)

def pickdata(image_dir,label_dir,des_dir):
    """
    将剩余的没有拼接pose的label移动到目标路径中
    Args:
        image_dir: 原始图像路径
        label_dir: 原始标签路径
        des_dir: 目标路径

    Returns:

    """
    os.makedirs(des_dir, exist_ok=True)
    des_img_dir = os.path.join(des_dir, 'images')
    des_lab_dir = os.path.join(des_dir, 'labels')
    os.makedirs(des_img_dir, exist_ok=True)
    os.makedirs(des_lab_dir, exist_ok=True)

    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(image_dir + "\\*." + extern))

    txtes = glob(des_lab_dir + '\\*.txt')
    txt_name = [i.replace("\\", "/").split("/")[-1].split(".txt")[0] for i in txtes]

    for image in tqdm(files):
        image_name = '.'.join(image.replace('\\', '/').split("/")[-1].split('.')[:-1])

        if image_name in txt_name:
            continue
        ori_label_path = os.path.join(label_dir,image_name+'.txt')
        shutil.copy(ori_label_path,des_lab_dir)
        shutil.copy(image,des_img_dir)

# pickdata(image_dir,label_dir,des_dir)

def get_last_label(des_dir):
    os.makedirs(des_dir, exist_ok=True)
    des_img_dir = os.path.join(des_dir, 'images')
    des_lab_dir = os.path.join(des_dir, 'labels')
    os.makedirs(des_img_dir, exist_ok=True)
    os.makedirs(des_lab_dir, exist_ok=True)

    txtes = glob(des_lab_dir + '\\*.txt')
    txt_name = [i.replace("\\", "/").split("/")[-1].split(".txt")[0] for i in txtes]

    for txt in tqdm(txtes):
        label_and_point = read_txt(txt)

        a = label_and_point[:,:5]
        b = label_and_point[:,6:]
        label = np.hstack((a,b))
        out_file = open(txt, 'w')
        for item in label:
            out_file.write(" ".join([str(a) for a in item[:]]) + '\n')

get_last_label(des_dir)
