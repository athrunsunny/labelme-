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
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils import data
from matplotlib import pyplot as plt
from scipy.integrate import simps
import matplotlib
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
        self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[203:206], dtype=np.float32)

        # img = letterbox(self.img, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        # img = self.img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)

        # if self.transforms:
        #     self.img = self.transforms(self.img)
        return (self.img, self.landmark, self.attribute, self.euler_angle,self.line[0])

    def __len__(self):
        return len(self.lines)


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

    source =r'G:\backup\project\PFLD-pytorch\data\test_data\list.txt'
    transform = transforms.Compose([transforms.ToTensor()])
    wlfw_val_dataset = WLFWDatasets(source, transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)

    nme_list = []
    pass_list = []
    # testing begin
    for img0, landmark_gt, _, _, p in tqdm(wlfw_val_dataloader):
        img_raw = cv2.imread(p[0], cv2.IMREAD_COLOR)

        img = np.float32(img_raw)
        landmark_gt = landmark_gt.to('cpu')
        im_height, im_width,_ = img.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
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
        landmark_points = list()
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            lmks = b[5:15]
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
            land = np.array(landmark_points) / scale1
            land = land.reshape(-1, 5, 2)
            nme_temp = compute_nme(land, landmark_gt)
            for item in nme_temp:
                nme_list.append(item)
        except:
            pass_list.append(p)


        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])

                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 1)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 1)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 1)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 1)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 1)
            # save image

            # name = "test.jpg"
            # cv2.imwrite(name, img_raw)
            # cv2.imshow('fff',img_raw)
            # cv2.waitKey(0)

    # nme
    print('nme: {:.4f}'.format(np.mean(nme_list)))
    # auc and failure rate
    failureThreshold = 0.1
    auc, failure_rate = compute_auc(nme_list, failureThreshold)
    print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
        failureThreshold, auc))
    print('failure_rate: {:}'.format(failure_rate))
    print('passing file :{},len: {}'.format(pass_list, len(pass_list)))

