
from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

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

def convert(size, box):
    box = np.float64(box)
    dw = np.float64(1. / (size[0]))
    dh = np.float64(1. / (size[1]))
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def convert_lmk(size, box):
    box = np.float64(box)
    dw = np.float64(1. / (size[0]))
    dh = np.float64(1. / (size[1]))
    box[::2] *= dw
    box[1::2] *= dh
    return box

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
from pathlib import Path
import glob
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # # Padded resize
        # img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        #
        # # Convert
        img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    source = r'G:\hp_tracking_proj\add_new_data\0221'
    dataset = LoadImages(source)
    for path, im, im0s, vid_cap, s in tqdm(dataset):


    # # testing begin
    # for i in range(1):
    #     image_path = "./curve/test_data0000007373.jpg"
    #     img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(im0s)

        im_height, im_width, _ = img.shape
        height, width, _ = img.shape
        size = (im_width,im_height)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)


        result_path = r'G:\backup\project\Pytorch_Retinaface1\results'
        image_name = path.replace('\\','/').split('/')[-1]

        image_n = image_name.split('.')[0]

        txt_path = r'G:\backup\project\Pytorch_Retinaface1\txt_results'
        save_txt_path = os.path.join(txt_path,image_n+'.txt')

        out_file = open(save_txt_path, 'w')
        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                # new_b = [b[0], b[2],b[1], b[3]]
                # padding = [-1.0, -1.0, -1.0]
                # b = b.tolist()
                # new_b = new_b + b[5:]
                # b = np.array(new_b)
                #
                # b[0:4] = convert(size, b[0:4])
                # b[4:] = convert_lmk(size, b[4:])

                b = b.tolist()
                b.pop(4)
                # b += padding
                # b = [0] + b
                b = np.array(b)
                annotation = np.zeros((1, 14))

                padding = [-1.0, -1.0, -1.0]
                label = b

                # bbox
                # 数据格式为x1y1x2y2
                label[0] = max(0, label[0])
                label[1] = max(0, label[1])
                label[2] = label[2] - label[0]
                label[3] = label[3] - label[1]

                label[2] = min(width - 1, label[2])
                label[3] = min(height - 1, label[3])
                annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
                annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
                annotation[0, 2] = label[2] / width  # w
                annotation[0, 3] = label[3] / height  # h

                # landmarks
                annotation[0, 4] = label[4] / width  # l0_x
                annotation[0, 5] = label[5] / height  # l0_y
                annotation[0, 6] = label[6] / width  # l1_x
                annotation[0, 7] = label[7] / height  # l1_y
                annotation[0, 8] = label[8] / width  # l2_x
                annotation[0, 9] = label[9] / height  # l2_y
                annotation[0, 10] = label[10] / width  # l3_x
                annotation[0, 11] = label[11] / height  # l3_y
                annotation[0, 12] = label[12] / width  # l4_x
                annotation[0, 13] = label[13] / height  # l4_y

                annotation = annotation.tolist()
                annotation = [2] + annotation[0] + padding

                b = annotation
                out_file.write(" ".join([str(p) for idx,p in enumerate(b)]) + '\n')
                b = list(map(int, b))
                cv2.rectangle(im0s, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(im0s, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(im0s, (b[5], b[6]), 1, (0, 0, 255), 1)
                cv2.circle(im0s, (b[7], b[8]), 1, (0, 255, 255), 1)
                cv2.circle(im0s, (b[9], b[10]), 1, (255, 0, 255), 1)
                cv2.circle(im0s, (b[11], b[12]), 1, (0, 255, 0), 1)
                cv2.circle(im0s, (b[13], b[14]), 1, (255, 0, 0), 1)
            # save image

            save_path = os.path.join(result_path,image_name)
            cv2.imwrite(save_path, im0s)

