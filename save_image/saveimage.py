import shutil

import cv2
import os
from glob import glob
from tqdm import tqdm
ROOT_DIR = os.getcwd()


class LoadUSBcam:
    """
    读取摄像头数据
    """
    INFO = ['fps', 'fw', 'fh', 'bs']

    def __init__(self, pipe='0', **options):
        # frameWidth = options.pop('frameWidth', 1280)
        # frameHeight = options.pop('frameHeight', 720)
        flip = options.pop('flip', False)
        bufferSize = options.pop('bufferSize', 10)
        running = options.pop('running', False)

        self.infoDict = dict()
        self.running = running
        # if pipe.isnumeric():
        #     pipe = eval(pipe)
        # self.pipe = pipe
        self.flip = flip

        # self.frameWidth = frameWidth
        # self.frameHeight = frameHeight
        self.bufferSize = bufferSize

        # self.cap = cv2.VideoCapture(self.pipe)
        # self.setprocess()
        # self.processeDict()

    def processeDict(self):
        self.infoDict['fps'] = self.fps
        self.infoDict['fw'] = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.infoDict['fh'] = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.infoDict['bs'] = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)

    def setprocess(self):
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.bufferSize)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))

    def __iter__(self):
        self.count = -1
        return self

    def __len__(self):
        return 0

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # 读取视频帧
        try:
            if self.pipe == 0:
                ret_val, frame = self.cap.read()
                if self.flip:
                    frame = cv2.flip(frame, 1)
            else:
                self.cap.grab()
                ret_val, frame = self.cap.retrieve()
                if self.flip:
                    frame = cv2.flip(frame, 1)
        except:
            raise StopIteration
        return frame

    def __getitem__(self, key):
        return self.infoDict[key]

    def getFrameCount(self):
        if isinstance(self.pipe, str):
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return int(self.count)

    def getFrameSize(self):
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return size

    def getFps(self):
        fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        return fps

    def getTime(self):
        if self.getFrameCount() == 0 or self.getFps() == 0:
            return 0
        videotime = round(self.getFrameCount() / self.getFps())
        return videotime

    def set(self, value):
        self.pipe = value
        self.cap = cv2.VideoCapture(self.pipe)
        self.setprocess()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


def detect_index(index):
    i = str(index)
    return len(i)


def process_image(video_path=ROOT_DIR):
    externs = ['mp4', 'avi',' ']
    files = list()
    for extern in externs:
        files.extend(glob(video_path + "\\*." + extern))
    print(files)
    image_savedir = os.path.join(video_path, 'create_frame')
    if not os.path.exists(image_savedir):
        os.makedirs(image_savedir)
    name = video_path.replace("\\", "/").split("/")[-1]
    loader = LoadUSBcam()
    count = 0
    for video in tqdm(files):
        print(video)
        video_name = video.replace("\\", "/").split("/")[-1].split(".")[0]

        save_path = os.path.join(image_savedir,video_name)
        os.makedirs(save_path, exist_ok=True)

        loader.set(video)
        framecount = loader.getFrameCount()
        print(framecount)
        for index, frame in enumerate(loader):
            try:
                length = detect_index(index)
                suffix = '0' * (8 - length) + str(index)
                if index == framecount:
                    break
                if index % 10 == 0:
                    # suffix = video_name + '_' + suffix + '.jpg'
                    suffix = suffix + '.jpg'
                    imgpath = os.path.join(save_path, suffix)
                    encoded_path = imgpath.encode("utf-8")
                    # cv2.imwrite(encoded_path, frame)
                    cv2.imencode('.jpg', frame)[1].tofile(encoded_path)
                count += 1
            except:
                pass
        loader.close()

if __name__ == '__main__':
    video = r'F:\lixb\43-412'
    process_image(video)
