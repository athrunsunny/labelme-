import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm


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
                '7--Cheering', '8--Election_Campain', '9--Press_Conference']
def trans_to():
    # image_dir = r'G:\hp_tracking_proj\train_iamges\publem\train\images'
    image_dir = r'G:\hp_tracking_proj\新标注规则标注的人体框'
    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(image_dir + "\\*\\*." + extern))

    image_name = [i.replace("\\", "/").split("/")[-1].split('.')[0] for i in files]
    assert len(files) == len(image_name)

    des_image_dir = r'G:\hp_tracking_proj\新标注规则标注的人头框'
    des_files = list()
    for extern in externs:
        des_files.extend(glob(des_image_dir + "\\*." + extern))

    des_image_name = [i.replace("\\", "/").split("/")[-1].split('.')[0] for i in des_files]

    count = 0
    trans_files = []
    for im_name in  tqdm(image_name):
        if im_name in des_image_name:
            continue
        im_prefix = im_name.split('_')[0]
        filename = ''
        for label in special_file:
            la = label.split('--')[0]
            if la == im_prefix:
            # if im_prefix in label:
                filename = label
                break

        image_path = os.path.join(image_dir,filename,im_name+'.jpg')
        shutil.copy(image_path,des_image_dir)
        count += 1
        trans_files.append(image_path)

    print('files:{},count:{}'.format(trans_files,count))

def trans_to_yolofile(src_path,des_path):
    # 将图片和标注的json文件复制到指定的文件夹下

    files = os.listdir(src_path)
    for file in tqdm(files):
        file_path = os.path.join(src_path,file)
        if not os.path.isdir(file_path):
            continue
        all_files = glob(file_path + "\\*")
        file_count = len(all_files)
        idx = 0
        for all_file in all_files:
            shutil.copy(all_file,des_path)
            idx += 1
        if file_count != idx:
            print('have some file not move')

if __name__ == '__main__':
    s_path = r'G:\hp_tracking_proj\新标注规则标注的人头框_验证集'
    d_path = r'G:\hp_tracking_proj\whead_yolotype________val'

    trans_to_yolofile(s_path,d_path)

