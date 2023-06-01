import os

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def get_part_image(mask_path, mask_pair_path, out_path):
    os.makedirs(out_path,exist_ok=True)
    # file_name = ['mask','mask_pair']
    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(mask_pair_path + "\\*." + extern))

    mask_files = list()
    for extern in externs:
        mask_files.extend(glob(mask_path + "\\*." + extern))

    mask_name = [i.replace("\\", "/").split("/")[-1] for i in mask_files]

    for img in tqdm(files):
        image_name = img.replace("\\", "/").split("/")[-1]
        real_name = '.'.join(image_name.split('.')[:-1])

        count = 0
        for mask in mask_files:
            if count > 20:
                continue
            mask_name = mask.replace("\\", "/").split("/")[-1]
            if mask_name[:len(real_name)] == real_name:
                output_dir = os.path.join(out_path, mask_name)
                output_dir = '.'.join(output_dir.split('.')[:-1]) + '.png'
                raw_image = cv2.imread(img)
                mask_image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

                mask_image[mask_image <= 30] = 0

                rows, cols = np.nonzero(mask_image)

                x1 = np.min(rows)
                x2 = np.max(rows)
                y1 = np.min(cols)
                y2 = np.max(cols)
                scenic_mask = ~mask_image
                mask_image = mask_image / 255.0
                raw_image[:, :, 0] = raw_image[:, :, 0] * mask_image
                raw_image[:, :, 1] = raw_image[:, :, 1] * mask_image
                raw_image[:, :, 2] = raw_image[:, :, 2] * mask_image

                new_image = raw_image[x1:x2, y1:y2, :]

                dst = cv2.cvtColor(new_image, cv2.COLOR_BGR2BGRA)
                new_mask = np.all(new_image[:, :, :] == [0, 0, 0], axis=-1)

                dst[new_mask, 3] = 0

                cv2.imwrite(output_dir, dst)
                count += 1


if __name__ == "__main__":
    out_path = r'D:\Users\yl3146\Desktop\tta\pa_\outputs\out'
    raw_path = r'D:\Users\yl3146\Desktop\tta\pa_\outputs\mask_pair'
    mask_path = r'D:\Users\yl3146\Desktop\tta\pa_\outputs\mask'

    get_part_image(mask_path, raw_path, out_path)
