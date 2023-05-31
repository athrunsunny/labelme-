import cv2
import numpy as np
ori_path = r'G:\backup\project\Grounded-Segment-Anything\outputs\mask_pair\000a318397212309577796457a836c2945d6c1ce.jpg'
# back_path = r'G:\backup\project\Grounded-Segment-Anything\outputs\000a318397212309577796457a836c2945d6c1ce.png'
mask_path = r'G:\backup\project\Grounded-Segment-Anything\outputs\mask\000a318397212309577796457a836c2945d6c1ce_mask_0.jpg'

person = cv2.imread(ori_path)
# back = cv2.imread(back_path)
#这里将mask图转化为灰度图
mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
#将背景图resize到和原图一样的尺寸
# back = cv2.resize(back,(person.shape[1],person.shape[0]))



mask[mask<=30] = 0

rows, cols = np.nonzero(mask)

x1 = np.min(rows)
x2 = np.max(rows)
y1 = np.min(cols)
y2 = np.max(cols)
# mask[mask>30] = 255.0
#这一步是将背景图中的人像部分抠出来，也就是人像部分的像素值为0
scenic_mask =~mask
# scenic_mask = scenic_mask  / 255.0
# back[:,:,0] = back[:,:,0] * scenic_mask
# back[:,:,1] = back[:,:,1] * scenic_mask
# back[:,:,2] = back[:,:,2] * scenic_mask
#这部分是将我们的人像抠出来，也就是背景部分的像素值为0
mask = mask / 255.0
person[:,:,0] = person[:,:,0] * mask
person[:,:,1] = person[:,:,1] * mask
person[:,:,2] = person[:,:,2] * mask

new_image = person[x1:x2,y1:y2,:]
#这里做个相加就可以实现合并
# result = cv2.add(back,person)

dst = cv2.cvtColor(new_image, cv2.COLOR_BGR2BGRA)
new_mask = np.all(new_image[:,:,:] == [0, 0, 0], axis=-1)

dst[new_mask,3] = 0

cv2.imwrite("3.jpg",new_image)
cv2.imwrite("dst.png", dst)

