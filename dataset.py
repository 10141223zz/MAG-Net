from torch.utils.data import Dataset
import torch
from torchvision import transforms as T
import os
from osgeo import gdal, osr
import numpy as np
import cv2
from torch.utils.data import DataLoader

img_w = 256
img_h = 256
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb

def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img

def big(xb,yb):
    affineShrink = np.array([[1.25, 0, 0], [0, 1.25, 0]], np.float32) 
    xb = cv2.warpAffine(xb, affineShrink, (100, 100), borderValue=125)
    yb = cv2.warpAffine(yb, affineShrink, (100, 100), borderValue=125)
    return xb,yb
    

def data_augment(xb,yb):

    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)

    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)

    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)

    # if np.random.random() < 0.25:
    #     xb,yb = rotate(xb,yb,45)

    # if np.random.random() < 0.25:
    #     xb,yb = rotate(xb,yb,135)

    # if np.random.random() < 0.25:
    #     xb,yb = rotate(xb,yb,315)

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 0)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 0)

    # if np.random.random() < 0.25:
    #     xb = cv2.flip(xb, -1)  # flipcode > 0：沿y轴翻转
    #     yb = cv2.flip(yb, -1)

#    if np.random.random() <0.25:
#        xb = random_gamma_transform(xb,1.0)

#    if np.random.random() < 0.25:
#        xb = blur(xb)

#    if np.random.random() < 0.25:
#        xb = add_noise(xb)

#    if np.random.random() < 0.25:
#        xb,yb = big(xb,yb)      

    return xb,yb


class C3dDataset(Dataset):
    def __init__(self,DataPath,DataType='train',InputSize=512):
        assert DataType in ("train", "val","test"), "Data type must be train or val or test"
        self.datatype = DataType
        self.image_size = (InputSize, InputSize)
        self.data_path = os.path.join(DataPath, DataType)
#        self.input_CLDAS = os.path.join(self.data_path,"input","CLDAS")
        self.input_remote = os.path.join(self.data_path,"input")
        self.label = os.path.join(self.data_path,"output")
        data_label = os.listdir(self.label)
        # print(data_label)
        data_input = os.listdir(self.input_remote)
        self.data_1 = [i.split('.')[0] for i in data_label]
        self.data_2 = [i.split('.')[0] for i in data_input]
#        print(self.data_2)
#        if (DataType == "train"):
#            self.input_transform = Compose([
#                # T.RandomHorizontalFlip(),
#                ToTensor(mean_list,notmal_list),
#                # Normalize(mean_list,
#                #           std_list)
#                # normalize
#            ])
#            # self.output_transform = Compose([
#            #     ToTensor(127,128),
#            #     # Normalize(0.5,0.2)
#            # ])
#        elif (DataType == "val"):
#            self.transform = T.Compose([
#                T.ToTensor(),
#                # normalize
#            ])

    def __getitem__(self, index):
        data_id = self.data_1[index]       #返回的是文件名前缀
        data_i = self.data_2[index]

        remote_path = os.path.join(self.input_remote, "{}.tif".format(data_i))
#        print(remote_path)

        try:
            img1 = gdal.Open(remote_path)
            im_width = img1.RasterXSize
            im_height = img1.RasterYSize
            # remote = img.ReadAsArray(0, 0)
            remote = img1.ReadAsArray(0,0,im_width,im_height)
            remote = np.transpose(remote,(1,2,0))
            # print(remote.shape)
        except:
            print(remote_path,index)
  
        label_path = os.path.join(self.label, "{}.png".format(data_id))
        label = cv2.imread(label_path,0)

        # if (self.datatype == "train"):
        #     remote, label = data_augment(remote,label)

        label=label[:, : ,np.newaxis]
        remote = np.transpose(remote,(2,0,1)).astype('float32')
        # print(label.shape)
        label = np.transpose(label,(2,0,1))
        # print(label.shape)
        seg_labels = np.zeros((2,512,512))
        label[label==0]=0
        label[label>0]=1
#        
        for c in range(2):
            seg_labels[c, :, :] = (label == c).astype(int)
        return data_i,remote, seg_labels.astype('float32')

    def __len__(self):
        return len(self.data_1)

if __name__ == '__main__':
    dataset = C3dDataset('./dataset',DataType='train')
# #    print(dataset.data)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    a = dataset[0]
    print(a)
    # print(dataset.data_1)
    # for  (x, y) in enumerate(train_loader):
        # print(x)
#    for iter, (x, y) in enumerate(train_loader):
#        print(iter)