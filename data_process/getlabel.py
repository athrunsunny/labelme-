import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm

# image_dir = r'G:\hp_tracking_proj\train_iamges\publem\train\images'
# image_dir = r'G:\backup\project\yolov5-multitask\data\widerface_head_body_pose_landmark_yolo1\train\images'
# label_dir = r'G:\backup\project\yolov5-multitask\data\widerface_head_body_pose_landmark_yolo1\train\labels'
image_dir = r'G:\hp_tracking_proj\three_identy_data\whead_yolotype________\train\images'
label_dir = r'G:\hp_tracking_proj\three_identy_data\whead_yolotype________\train\labels'
# image_dir = r'G:\hp_tracking_proj\three_identy_data\test\createjson\train\images'
# label_dir = r'G:\hp_tracking_proj\three_identy_data\test\createjson\train\labels'
externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
files = list()
for extern in externs:
    files.extend(glob(image_dir + "\\*." + extern))

image_name = [i.replace("\\", "/").split("/")[-1].split('.')[0] for i in files]
assert len(files) == len(image_name)


def read_ori_head(image_name):
    head_label = list()
    # for txt_name in image_name:
    labeltxt = os.path.join(label_dir, image_name + '.txt')
    if not os.path.isfile(labeltxt):
        return [],0
    head_label_txt = open(labeltxt, 'r')
    lines = head_label_txt.readlines()

    for line in lines:
        if line[0] == '0.0' or line[0] == '0':
            # line = line.split(' ')[:-4]
            # 保持元数据的对齐
            padding = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
            # 从之前的三类数据集上需要加[:-1]
            # line = line.split(' ')[:-1]
            line = line.split(' ')
            label = [float(x) for x in line if x != '']
            label[0] = int(label[0])
            head_label.append(label + padding)

    read_time = 0
    if head_label:
        read_time = 1
    return head_label, read_time


# person_label_dir = r'G:\hp_tracking_proj\widerface_p\train\labels'
# person_label_dir = r'G:\hp_tracking_proj\widerface_just_person_yolo_type\train_allpic\labels'
person_label_dir = r'G:\hp_tracking_proj\three_identy_data\wperson_yolotype________\train\labels'
def read_person(image_name):
    person_label = list()

    # for txt_name in image_name:
    labeltxt = os.path.join(person_label_dir, image_name + '.txt')
    if not os.path.isfile(labeltxt):
        return [],0
    person_label_txt = open(labeltxt, 'r')
    lines = person_label_txt.readlines()

    for line in lines:
        if line[0] == '0.0' or line[0] == '0':
            # padding = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
            # 保持元数据的对齐
            padding = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
            line = line.split(' ')
            label = [float(x) for x in line if x != '']
            label[0] = int(label[0] + 1)
            person_label.append(label + padding)
    read_time = 0
    if person_label:
        read_time = 1
    return person_label, read_time


# face_label_dir = r'G:\hp_tracking_proj\WIDER_train\change\labels'
face_label_dir = r'G:\hp_tracking_proj\three_identy_data\wface_yolotype________\train\labels'

def read_oir_face_lmk(image_name):
    face_label = list()

    # for txt_name in image_name:
    labeltxt = os.path.join(face_label_dir, image_name + '.txt')
    if not os.path.isfile(labeltxt):
        return [],0
    face_label_txt = open(labeltxt, 'r')
    lines = face_label_txt.readlines()

    for line in lines:
        if line[0] == '0.0' or line[0] == '0':
            # line = line.split(' ')[:-2]
            # 保持元数据的对齐
            padding = [-1.0, -1.0, -1.0]
            line = line.split(' ')
            label = [float(x) for x in line if x != '']
            if label[1] < 0 or label[2] < 0 or label[3] <= 0 or label[4] <= 0:
                print(image_name,label)
                continue
            label[0] = int(label[0] + 2)
            face_label.append(label + padding)
    read_time = 0
    if face_label:
        read_time = 1
    return face_label, read_time


train_image_des_dir = r'G:\hp_tracking_proj\three_identy_data\union\images'
if not os.path.exists(train_image_des_dir):
    os.makedirs(train_image_des_dir)

label_image_des_dir = r'G:\hp_tracking_proj\three_identy_data\union\labels'
if not os.path.exists(label_image_des_dir):
    os.makedirs(label_image_des_dir)


def create_last_label(image_name):
    for im_name in tqdm(image_name):
        # try:
        #     image_n = f'{image_dir}/{im_name}.jpg'
        #     out_file = open('%s/%s.txt' % (label_image_des_dir, im_name), 'w')
        #     head_label, h_time = read_ori_head(im_name)
        #     person_label, p_time = read_person(im_name)
        #     face_label, f_time = read_oir_face_lmk(im_name)
        #
        #     for label in head_label:
        #         if len(label) != 0:
        #             out_file.write(" ".join([str(p) for p in label]) + '\n')
        #     for label in person_label:
        #         if len(label) != 0:
        #             out_file.write(" ".join([str(p) for p in label]) + '\n')
        #     for label in face_label:
        #         if len(label) != 0:
        #             out_file.write(" ".join([str(p) for p in label]) + '\n')
        #
        #     total_label = head_label + person_label + face_label
        #     if len(total_label) == 0:
        #         print(image_n)
        #     shutil.copy(image_n, train_image_des_dir)
        # except:
        #     pass
        image_n = f'{image_dir}/{im_name}.jpg'
        out_file = open('%s/%s.txt' % (label_image_des_dir, im_name), 'w')
        head_label, h_time = read_ori_head(im_name)
        person_label, p_time = read_person(im_name)
        face_label, f_time = read_oir_face_lmk(im_name)

        for label in head_label:
            if len(label) != 0:
                out_file.write(" ".join([str(p) for p in label]) + '\n')
        for label in person_label:
            if len(label) != 0:
                out_file.write(" ".join([str(p) for p in label]) + '\n')
        for label in face_label:
            if len(label) != 0:
                out_file.write(" ".join([str(p) for p in label]) + '\n')

        total_label = head_label + person_label + face_label
        if len(total_label) == 0:
            print(image_n)
        shutil.copy(image_n, train_image_des_dir)


if __name__ == '__main__':
    create_last_label(image_name)
