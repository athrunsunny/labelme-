import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('AGG')

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list, name):
    value = 0  # 0 for background

    # mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img = torch.zeros(mask_list.shape[-2:])
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{name}_mask_{idx}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    # json_data = [{
    #     'value': value,
    #     'label': 'background'
    # }]
    # for label, box in zip(label_list, box_list):
    #     value += 1
    #     name, logit = label.split('(')
    #     logit = logit[:-1] # the last is ')'
    #     json_data.append({
    #         'value': value,
    #         'label': name,
    #         'logit': float(logit),
    #         'box': box.numpy().tolist(),
    #     })
    # with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
    #     json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str,
                        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', )  # required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default='groundingdino_swint_ogc.pth',
        # required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default='sam_vit_h_4b8939.pth',  # required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, default='test', )  # required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, default='animal', )  # required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs",  # required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(image_path + "\\*." + extern))

    for ind,img in enumerate(tqdm(files)):
        if ind < 78:
            continue
        try:
            image_path = img
            image_name = img.replace("\\", "/").split("/")[-1]
            real_name = ".".join(image_name.split('.')[:-1])

            # load image
            image_pil, image = load_image(image_path)
            # load model
            model = load_model(config_file, grounded_checkpoint, device=device)

            raw_image_path = os.path.join(output_dir, 'raw_image')
            os.makedirs(raw_image_path, exist_ok=True)

            # visualize raw image
            image_pil.save(os.path.join(raw_image_path, f"{real_name}_raw.jpg"))

            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, device=device
            )

            # initialize SAM
            predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )

            mask_pair_path = os.path.join(output_dir, 'mask_pair')
            os.makedirs(mask_pair_path, exist_ok=True)

            mask_path = os.path.join(output_dir, 'mask')
            os.makedirs(mask_path, exist_ok=True)
            # draw output image
            # plt.figure(figsize=(10, 10))
            plt.figure()
            # plt.imshow(image)
            for idx, mask in enumerate(masks):
                if idx == 0:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    # show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                    # for box, label in zip(boxes_filt, pred_phrases):
                    #     show_box(box.numpy(), plt.gca(), label)

                    plt.axis('off')
                    plt.savefig(
                        os.path.join(mask_pair_path, f"{real_name}.jpg"),
                        bbox_inches="tight", dpi=300, pad_inches=0.0
                    )

                save_mask_data(mask_path, masks, boxes_filt, pred_phrases, real_name)
        except:
            pass
