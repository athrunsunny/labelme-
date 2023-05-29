import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import torch.nn.functional as F
from torchvision import transforms
from model.model import Model
from utils.torch_utils import select_device
from torch.utils.data import DataLoader
from torch.utils import data
from matplotlib import pyplot as plt
import matplotlib
from scipy.integrate import simps
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from model.landmark.net_nano import create_net
from utils.dataloader import LoadImages
from utils.general import colorstr, LOGGER, check_yaml, print_args, increment_path, Profile, scale_boxes, \
    scale_coords_landmarks, letterbox
from utils.landmark.box_utils import PriorBox, decode_landm, decode, py_cpu_nms


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pth', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'imgs', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str,
                        default=r'G:\backup\project\yolofamily\0208\yolov5multask_with_norm\val_landmark_data\test_data/list.txt',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--cfg', type=str, default=ROOT / 'model/mobile_nano.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-lmk.yaml', help='hyperparameters path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=112,
                        help='inference size w,h')
    parser.add_argument('--conf-thres', type=float, default=0.02, help='confidence threshold')
    parser.add_argument('--max-det', type=int, default=1500, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default=ROOT / 'runs/detect-lmk', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


class WLFWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None, img_size=640, stride=32, auto=True):
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.line = None
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])
        self.path = self.line[0]
        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)
        self.attribute = np.asarray(self.line[197:203], dtype=np.float32)
        self.euler_angle = np.asarray(self.line[203:206], dtype=np.float32)

        img = letterbox(self.img, self.img_size, auto=self.auto)[0]

        def padding_img(img, img_size, color=(114, 114, 114)):
            im = img
            shape = im.shape[:2]
            X = torch.tensor(im).transpose(0, 2).transpose(2, 1)

            new_shape = (img_size[0] - shape[0]) // 2
            # print("shape:", X.shape)
            dim = (new_shape, new_shape, new_shape, new_shape)  # left,right,top ,down
            X = F.pad(X, dim, "constant", value=114).transpose(2, 1).transpose(0, 2)
            padX = X.data.numpy()

            top, bottom = 0, 0
            left, right = 0, 0
            X = cv2.copyMakeBorder(padX, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

            return X

        # img = padding_img(self.img, self.img_size)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # if self.transforms:
        #     self.img = self.transforms(self.img)
        return (img, self.landmark, self.attribute, self.euler_angle, self.path)

    def __len__(self):
        return len(self.lines)


def padding_img(img, img_size, color=(114, 114, 114)):
    im = img
    shape = im.shape[:2]
    X = torch.tensor(im).transpose(0, 2).transpose(2, 1)

    new_shape = (img_size[0] - shape[0]) // 2
    # print("shape:", X.shape)
    dim = (new_shape, new_shape, new_shape, new_shape)  # left,right,top ,down
    X = F.pad(X, dim, "constant", value=114).transpose(2, 1).transpose(0, 2)
    padX = X.data.numpy()

    top, bottom = 0, 0
    left, right = 0, 0
    X = cv2.copyMakeBorder(padX, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return X


def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8,] - pts_gt[9,])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36,] - pts_gt[45,])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60,] - pts_gt[72,])
        else:
            # raise ValueError('Number of landmarks is wrong')
            interocular = np.linalg.norm(pts_gt[0,] - pts_gt[1,])
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt,
                                        axis=1)) / (interocular * L)

    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=True):
    matplotlib.use('TkAgg')
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate


def compute_5point(lmks):
    new_lmks = list()
    lmks = lmks.reshape(lmks.shape[0], -1, 2).cpu().numpy()  # landmark_gt
    shape = lmks.shape[0]
    lmks = lmks.reshape(-1, 2)  # landmark_gt
    new_lmks.append(lmks[96, :])
    new_lmks.append(lmks[97, :])
    new_lmks.append(lmks[54, :])
    new_lmks.append(lmks[76, :])
    new_lmks.append(lmks[82, :])
    new_lmks = np.array(new_lmks)
    new_lmks = new_lmks.reshape(shape, -1, 2)
    return new_lmks


def postprocess(loc, conf, landms, size, resize, top_k=-1, keep_top_k=-1, nms_threshold=0.1, conf_thres=0.25,
                device=None, hyp=None):
    height, width = size
    scale = torch.Tensor([width, height, width, height])
    scale = scale.to(device)
    conf = F.softmax(conf, dim=2)
    priorbox = PriorBox(image_size=(height, width), hyp=hyp)
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, [hyp['CENTER_VAR'], hyp['SIZE_VAR']])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()

    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, [hyp['CENTER_VAR'], hyp['SIZE_VAR']])
    scale1 = torch.Tensor([width, height, width, height, width, height, width, height, width, height])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > conf_thres)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    return dets


def detect(
        weights=ROOT / '',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        hyp=ROOT / 'data/hyps/hyp.scratch-lmk.yaml',
        cfg=ROOT / 'model/mobile_nano.yaml',
        imgsz=320,  # inference size
        conf_thres=0.05,  # confidence threshold
        max_det=1500,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        origin_size=False,  # use origin image size to evaluate
        top_k=5000,
        nms_threshold=0.4,
        vis_thres=0.6,
):
    source = str(source)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    hyp = check_yaml(hyp)
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    hyp = hyp.copy()

    # Load model
    weights = str(weights)
    onnx_model = weights.endswith('.onnx')
    if onnx_model:
        import onnxruntime
        net = onnxruntime.InferenceSession(weights)
        input_name = net.get_inputs()[0].name
        input_size = net.get_inputs()[0].shape
        onnx_input_size = input_size[2:]
        if imgsz != onnx_input_size:
            LOGGER.info(colorstr('size info: ') + f'Your input imgsz {imgsz},but onnx get input size {onnx_input_size},'
                                                  f'Now change input size to {onnx_input_size}')
            imgsz = onnx_input_size
    else:
        # net = create_net()
        # net.load(weights)
        # net.eval()

        cfg = check_yaml(cfg)
        device = select_device(device)
        net = Model(cfg, ch=3, nc=1).to(device)
        pretrained_dict = torch.load(str(weights), map_location='cpu')
        net.load_state_dict(pretrained_dict, strict=False)
        net.eval()
        net = net.fuse()

    LOGGER.info(colorstr('image size: ') + f'{imgsz}')

    # dataset = LoadImages(source, img_size=imgsz, oresize=origin_size, onnx=onnx_model, auto=False)

    bs = 1  # batch_size
    transform = transforms.Compose([transforms.ToTensor()])
    wlfw_val_dataset = WLFWDatasets(source, transform, img_size=imgsz)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)

    total = 0
    resize = 1
    nme_list = []
    pass_list = []
    dt = (Profile(), Profile(), Profile())
    # for path, im, im0s, vid_cap, s, resize in dataset:
    for img, landmark_gt, _, _, p in tqdm(wlfw_val_dataloader):
        with dt[0]:
            img_raw = cv2.imread(p[0], cv2.IMREAD_COLOR)
            im0s = img_raw.copy()
            new_shape = 0
            if 0:
                img_raw = padding_img(img_raw, (640, 640))
                new_shape = (640 - img0.shape[1]) // 2

            img = np.float32(img_raw)
            landmark_gt = landmark_gt.to('cpu')
            im_height, im_width, _ = img.shape
            scale_lmk = torch.Tensor(
                [im_width, im_height, im_width, im_height, im_width, im_height, im_width, im_height, im_width,
                 im_height])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale_lmk = scale_lmk.to(device)
            images = img

        # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # im.jpg
        # with dt[0]:
        #     images = torch.from_numpy(im).to(device)
        #     if len(images.shape) == 3:
        #         images = images[None]
        #     if onnx_model:
        #         images = images.numpy()
        #         images = images.astype(np.float32)
        #     else:
        #         images = images.to(device).float()

        with dt[1]:
            with torch.no_grad():
                if onnx_model:
                    boxes, scores, landmarks = net.run(None, {input_name: images})
                    boxes, scores, landmarks = torch.as_tensor(boxes), torch.as_tensor(scores), torch.as_tensor(
                        landmarks)
                else:
                    boxes, scores, landmarks = net(images)
        with dt[2]:
            dets = postprocess(boxes, scores, landmarks, images.shape[2:], resize, top_k=top_k, keep_top_k=max_det // 2,
                               nms_threshold=nms_threshold, conf_thres=conf_thres, hyp=hyp)
            if onnx_model:
                dets = torch.as_tensor(dets)
                dets[:, :4] = scale_boxes(images.shape[2:], dets[:, :4], im0.shape).round()
                dets[:, 5:15] = scale_coords_landmarks(images.shape[2:], dets[:, 5:15], im0.shape).round()
                dets = dets.numpy()
        LOGGER.info(f"{'' if dets.shape[0] else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        landmark_points = list()
        for b in dets:
            if b[4] < vis_thres:
                continue
            lmks = b[5:15] - np.array(new_shape)
            landmark_points.append(lmks)
        dets = dets[dets[:, 4] > 0.6]
        if dets.shape[0] > 1:
            continue
        if dets.shape[0] > 1:
            max_area = -1
            idx = -1
            for j in range(dets.shape[0]):
                xyxy = dets[j, :4].tolist()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > max_area:
                    max_area = area
                    idx = j
            landmark_points = landmark_points[idx]
        try:
            landmark_gt = compute_5point(landmark_gt)

            if 0:
                scale_lmk = torch.Tensor([img0.shape[1], img0.shape[2], img0.shape[1], img0.shape[2],
                                          img0.shape[1], img0.shape[2], img0.shape[1], img0.shape[2],
                                          img0.shape[1], img0.shape[2]])

            land = np.array(landmark_points) / scale_lmk
            land = land.reshape(-1, 5, 2)
            nme_temp = compute_nme(land, landmark_gt)
            for item in nme_temp:
                nme_list.append(item)
        except:
            pass_list.append(p)

        count_face = 0
        for b in dets:
            if b[4] < vis_thres:
                continue
            count_face += 1
            probs = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # cv2.rectangle(im0s, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            # cx = b[0]
            # cy = b[1] + 12
            # cv2.putText(im0s, probs, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # lmks
            cv2.circle(im0s, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(im0s, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(im0s, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(im0s, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(im0s, (b[13], b[14]), 1, (255, 0, 0), 4)

        # save images
        save_path = r'G:\backup\project\UF\for_test\Light-Face-Detector-with-Landmark--\test_nme'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_name = p[0].replace("\\", "/").split("/")[-1]
        name = os.path.join(save_path, img_name)
        cv2.imwrite(name, im0s)
        if view_img:
            cv2.imshow('result', im0s)
            cv2.waitKey(0)
        total += count_face
        LOGGER.info(f"Found {count_face} faces. The output image is {save_path}")

    # nme
    LOGGER.info('nme: {:.4f}'.format(np.mean(nme_list)))
    # auc and failure rate
    failureThreshold = 0.1
    auc, failure_rate = compute_auc(nme_list, failureThreshold)
    LOGGER.info('auc @ {:.1f} failureThreshold: {:.4f}'.format(
        failureThreshold, auc))
    LOGGER.info('failure_rate: {:}'.format(failure_rate))
    LOGGER.info('passing file :{},len: {}'.format(pass_list, len(pass_list)))

    LOGGER.info(f"Total face {total}")


if __name__ == "__main__":
    opt = parse_opt()
    detect(**vars(opt))
