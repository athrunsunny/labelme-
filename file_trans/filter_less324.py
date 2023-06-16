import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm



def get_files(path, exts=('jpg', 'png', 'bmp'), recursive=False):
    import glob
    files = []
    for ext in exts:
        if recursive:
            files.extend(glob.iglob(os.path.join(path, '**/*.{}'.format(ext)), recursive=True))
        else:
            files.extend(glob.glob(os.path.join(path, '*.{}'.format(ext))))
    return files


def remove_small_obj(im_path, lb_path, out_path,output_im_path, area_thr=225, target_size=640):
    """
    
    :param im_path: 
    :param lb_path: cls CxCyWH  归一化后的坐标
    :param out_path: 
    :param area_thr: 过滤面积阈值
    :return:
    """
    os.makedirs(output_im_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    im_files = get_files(im_path)

    cnt = 0
    for f_im in tqdm(im_files):
        # if i % 100 == 0:
        #     print(i, f_im)
        basename = os.path.splitext(os.path.basename(f_im))[0]
        im = cv2.imread(f_im, cv2.IMREAD_COLOR)
        im_h, im_w = im.shape[:2]

        f_lb = os.path.join(lb_path, basename + '.txt')
        if not os.path.isfile(f_lb): continue

        f_lb_out = os.path.join(out_path, basename + '.txt')
        f_out_txt = open(f_lb_out, 'a')
        with open(f_lb, 'r') as f_txt:
            lines = f_txt.read().strip().splitlines()

            for line in lines:
                line = line.split()
                cls = line[0]
                box = [float(e) for e in line[1: 5]]

                box_w, box_h = box[2] * im_w, box[3] * im_h
                if box_w < 0 or box_h < 0:
                    # print(box_w, box_h)
                    print('error', basename)

                resize_ratio = target_size / max(im_h, im_w)
                if box_w * box_h * resize_ratio * resize_ratio < area_thr:
                    cnt += 1
                    continue

                if (cls == '1' or cls == '1.0') and box_w * box_h < 2 * area_thr:
                    cnt += 1
                    continue

                box = [e for e in line]
                # box.insert(0, cls)

                f_out_txt.writelines(' '.join(box) + '\n')
        f_out_txt.close()
        # shutil.copy(f_im,out_path_image)
    print('remove objs: ', cnt)


if __name__ == '__main__':
    im_path = r'G:\hp_tracking_proj\three_identy_data\2690_val_dataset\val\images'
    lb_path = r'G:\hp_tracking_proj\three_identy_data\2690_val_dataset\val\labels'
    out_path = r'G:\hp_tracking_proj\three_identy_data\widerface_head_body_pose_lmk_move_1212_fix_sit_szclean_320x320\val\labels'
    out_path_image = r'G:\hp_tracking_proj\three_identy_data\widerface_head_body_pose_lmk_move_1212_fix_sit_szclean_320x320\val\images'
    remove_small_obj(im_path, lb_path, out_path,out_path_image, area_thr=324, target_size=640)
