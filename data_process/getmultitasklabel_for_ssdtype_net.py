import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm


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


def read_txt(path):
    assert os.path.exists(path)
    with open(path, mode='r', encoding="utf-8") as f:
        content = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float64)
    res = np.unique(content, axis=0)
    if len(res) == 0:
        return None
    return res


# image_dir = r'G:\backup\project\yolov5-multitask\data\widerface_head_body_pose_landmark_yolo\train\images'
# label_dir = r'G:\hp_tracking_proj\widerface_head_body_pose_lmk\val\labels'
label_dir = r'G:\hp_tracking_proj\three_identy_data\widerface_head_body_pose_lmk_move_1212_fix_sit_szclean\train\labels'
label_image_des_dir = r'G:\hp_tracking_proj\three_identy_data\widerface_ssd_type_net_datasets'
# externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
# files = list()
# for extern in externs:
#     files.extend(glob(image_dir + "\\*." + extern))
#
# image_name = [i.replace("\\", "/").split("/")[-1].split('.')[0] for i in files]
# assert len(files) == len(image_name)


special_file = ['0--Parade', '1--Handshaking', '10--People_Marching', '11--Meeting', '12--Group', '13--Interview',
                '14--Traffic', '15--Stock_Market', '16--Award_Ceremony', '17--Ceremony', '18--Concerts', '19--Couple',
                '2--Demonstration', '20--Family_Group', '21--Festival', '22--Picnic', '23--Shoppers',
                '24--Soldier_Firing', '25--Soldier_Patrol', '26--Soldier_Drilling', '27--Spa', '28--Sports_Fan',
                '29--Students_Schoolkids', '3--Riot', '30--Surgeons', '31--Waiter_Waitress', '32--Worker_Laborer',
                '33--Running', '34--Baseball', '35--Basketball', '36--Football', '37--Soccer', '38--Tennis',
                '39--Ice_Skating', '4--Dancing', '40--Gymnastics', '41--Swimming', '42--Car_Racing', '43--Row_Boat',
                '44--Aerobics', '45--Balloonist', '46--Jockey', '47--Matador_Bullfighter',
                '48--Parachutist_Paratrooper', '49--Greeting', '5--Car_Accident', '50--Celebration_Or_Party',
                '51--Dresses', '52--Photographers', '53--Raid', '54--Rescue', '55--Sports_Coach_Trainer', '56--Voter',
                '57--Angler', '58--Hockey', '59--people--driving--car', '6--Funeral', '61--Street_Battle',
                '7--Cheering', '8--Election_Campain', '9--Press_Conference',

                '62--office_station','63--mosaic_pad','64--wider_val','65--sit','66--blackscreen','67--half_body',
                '68--office_ten','69--mtfl_lfw','70--overexposure','71--mix_condition_refine'

                ]


def create_ssd_type_label(label_dir, label_image_des_dir):
    if not os.path.exists(label_image_des_dir):
        os.makedirs(label_image_des_dir)

    files = glob(label_dir + "\\*.txt")
    image_dir = label_dir.replace('labels', 'images')

    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    imfiles = list()
    for extern in externs:
        imfiles.extend(glob(image_dir + "\\*." + extern))
    im_names = [i.replace("\\", "/").split("/")[-1] for i in imfiles]

    out_file = open('%s/%s.txt' % (label_image_des_dir, 'hpf_label_'), 'w')
    out_file_name = ''
    for filename in tqdm(files):
        txt_name = filename.replace("\\", "/").split("/")[-1]
        tmp = txt_name.split('_')
        file_num, file_name = tmp[0], tmp[1]

        for name in special_file:
            name_num = name.split('--')[0]
            if name_num == file_num:
                out_file_name = name
                break

        out_file_path = os.path.join(label_image_des_dir, 'images', out_file_name)
        if not os.path.exists(out_file_path):
            os.makedirs(out_file_path)

        out_file_name = out_file_name + "/"
        image_name = txt_name.split('.')[0] + '.jpg'
        save_file_name = "# " + out_file_name + image_name + "\n"
        if image_name in im_names:
            image_path = filename.replace('labels', 'images').replace('.txt', '.jpg')
            frame = cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            size = (width, height)
            # shutil.copy(image_path, out_file_path)

        out_file.write(save_file_name)
        label_and_point = read_txt(filename)
        label_and_point = label_and_point[label_and_point[:, 0] == 2][:, :15]
        label_and_point[:, 1:5] = reconvert_np(size, label_and_point[:, 1:5])
        label_and_point[:, 5:15] = reconvert_np_lmk(size, label_and_point[:, 5:15])
        label_and_point = label_and_point[:, 1:]

        anno = np.zeros([label_and_point.shape[0],20])
        anno[:,:4] = label_and_point[:,:4]
        anno[:,4:6] = label_and_point[:,4:6]
        anno[:, 7:9] = label_and_point[:, 6:8]
        anno[:, 10:12] = label_and_point[:, 8:10]
        anno[:, 13:15] = label_and_point[:, 10:12]
        anno[:, 16:18] = label_and_point[:, 12:14]
        label_and_point = anno

        for length in range(label_and_point.shape[0]):
            str_label = "0 "
            for i in range(len(label_and_point[0])):
                str_label = str_label + " " + str(label_and_point[length][i])
            str_label = str_label.replace('[', '').replace(']', '')
            str_label = str_label.replace(',', '') + '\n'
            str_label = str_label[3:]
            out_file.write(str_label)


if __name__ == '__main__':
    create_ssd_type_label(label_dir, label_image_des_dir)
