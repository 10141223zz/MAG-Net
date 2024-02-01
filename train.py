from __future__ import print_function
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
import torch.nn as nn
from osgeo import gdal, osr
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from GMRR26 import *
from dataset import *
import pandas as pd
from matplotlib import pyplot as plt
import time
import sys
# import focal_loss as fcl
import types
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import skimage.io
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,roc_curve, auc,roc_auc_score,log_loss,confusion_matrix
import warnings
from torch.nn.utils import clip_grad_norm_
from pytorch_msssim import ms_ssim, ssim

warnings.filterwarnings("ignore")

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def visualize_feature_map(img_batch):
    feature_map = img_batch
#    print(feature_map.shape)
    feature_map_combination = []
    plt.figure(figsize=(50, 32))
    num_pic = feature_map.shape[2]
    # print(num_pic)
    row, col = get_row_col(num_pic)
    # print(row, col)
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
#        plt.subplots_adjust(wspace =1, hspace =1)
        plt.imshow(feature_map_split)
        plt.axis('off')
        plt.title('feature_map_{}'.format(i))
    plt.savefig('feature_map.png')
    plt.show()
    feature_map_sum = sum(ele for ele in feature_map_combination)
#    print(feature_map_sum)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")

n_class    = 2
batch_size = 2
epochs     = 60
lr         = 1e-5
momentum   = 0.001
w_decay    = 1e-4   # 正则化作用
step_size  = 30
gamma      = 0.1


configs = "batch{}_epoch{}.pth".format(batch_size, epochs)
# configs    = "GMRR1_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

# create dir for model
model_dir = "GMRR26_1"


if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

train_data =C3dDataset('./dataset',DataType='train')
val_data =C3dDataset('./dataset',DataType='val')
test_data = C3dDataset('./dataset',DataType='test')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

psp_model = unet()

if use_gpu:
    ts = time.time()
    psp_model = psp_model.cuda()
#    fcn_model = fcn_model.cuda()
#    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

# class dice_bce_loss(nn.Module):
#     def __init__(self, batch=True):
#         super(dice_bce_loss, self).__init__()
#         self.batch = batch
#         self.bce_loss = nn.BCEWithLogitsLoss()
#     def soft_dice_coeff(self, y_pred, y_true):
#         smooth = 0.0  # may change
#         if self.batch:
#             i = torch.sum(y_true)
#             j = torch.sum(y_pred)
#             intersection = torch.sum(y_true * y_pred)
#         else:
#             i = y_true.sum(1).sum(1).sum(1)
#             j = y_pred.sum(1).sum(1).sum(1)
#             intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
#         score = (2. * intersection + smooth) / (i + j + smooth)
#
#         # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
#         return score.mean()
#     def soft_dice_loss(self, y_pred, y_true):
#         loss = 1 - self.soft_dice_coeff(y_pred, y_true)
#         return loss
#     def __call__(self, y_pred, y_true):
#         # y_true =y_true.detach()
#         # y_p = Variable(y_pred,requires_grad=True)
#         # print(y_p.requires_grad,y_true.requires_grad)
#         a = self.bce_loss(y_pred, y_true)
#         # print(y_pred.requires_grad,y_true.requires_grad)
#         b = self.soft_dice_loss(F.sigmoid(y_pred), y_true )
#
#         return a+b, a

# criterion = dice_bce_loss()
criterion = nn.BCEWithLogitsLoss()
criterion1 = ssim

optimizer = optim.RMSprop(psp_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
# optimizer = optim.Adam(psp_model.parameters(), lr=lr, eps=1e-8, weight_decay=w_decay)

scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
# create dir for score
score_dir = os.path.join("scores", configs)

if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores   = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)

def train():
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s6 = []
    s7 = []
    s8 = []
    s9 = []
    s10 = []
    for epoch in range(epochs):
        print('epoch:', epoch)
        train_loss = 0
        acc_sum = 0
        r_sum = 0
        p_sum = 0
        IOU_sum = 0
        f1_sum = 0
        num = 0
        ts = time.time()
        psp_model.train()
        for iter, (q, x, y) in enumerate(train_loader):
            # print('iter:',iter)
            x, y_true = Variable(x), Variable(y)  # tensor
            inputs = x.cuda()
            labels = y_true.cuda()
            outputs = psp_model(inputs)
            loss= criterion(outputs, labels)

            loss_sum = loss
            # print(loss_sum)
            optimizer.zero_grad()
            loss_sum.backward()
            # clip_grad_norm_(psp_model.parameters(), max_norm=12, norm_type=2)
            optimizer.step()
            # print('loss.item:',loss.item())
            train_loss += loss_sum.item()
            if iter % 100 == 0:
                print("epoch{}, iter{},loss: {}".format(epoch, iter, loss_sum.item()))  # 输出的是第50倍数的loss

        torch.save(psp_model, model_path)

        # save_file = {"model": psp_model.state_dict()}
        # torch.save(save_file, model_path)

        print('train_loss:', train_loss)
        s1.append(train_loss / len(train_loader))
        y1 = s1
        y1 = pd.DataFrame(y1)
        w1 = pd.ExcelWriter('train_loss_GMRR26_1.xlsx')  # 写入Excel文件
        y1.to_excel(w1, 'train_loss', float_format='%.16f')
        w1.save()
        w1.close()
        #    total_ious = []
        #    pixel_accs = []
        psp_model.eval()
        val_loss_512 = 0
        val_loss_64 = 0
        val_loss_256 = 0
        val_loss_sum = 0

        for iter, (a, x, y) in enumerate(val_loader):
            x_val, y_true_val = x, y
            inputs_val = x_val.cuda()
            labels_val = y_true_val.cuda()
            outputs = psp_model(inputs_val)

            loss = criterion(outputs, labels_val)

            val_loss_512 += loss.item()
            val_loss_sum += loss.item()

# 为了曲线向上平移
#             loss = loss0 + loss1 + loss2
#             val_loss_512 += loss0.item()
#             val_loss_64 += loss1.item()
#             val_loss_256 += loss2.item()
#             val_loss_sum += loss.item()


            # 评价指标
            outputs_val = outputs.data.cpu().numpy()
            # print(outputs_val)
            # print(outputs_val.shape)
            outputs_val = np.squeeze(outputs_val)
            outputs_val = np.transpose(outputs_val, (1, 2, 0))
            # print(outputs_val)
            pr = outputs_val.reshape((512, 512, 2)).argmax(axis=2)
            # print(pr.shape)
            array_pr = np.reshape(pr, -1)
            # print(array_pr)

            lab = y_true_val.data.cpu().numpy()
            # print(lab)
            # print(lab.shape) #(1, 2, 512, 512)
            lab = np.squeeze(lab)
            # print(lab)
            # print(lab.shape) #(2, 512, 512)
            lab = np.transpose(lab, (1, 2, 0))
            # print(lab)
            # print(lab.shape) #(512, 512, 2)
            la = lab.reshape((512, 512, 2)).argmax(axis=2)
            # print(la)
            # print(la.shape) #(512, 512)
            array_la = np.reshape(la, -1)
            acc = pixel_acc(array_la, array_pr)
            p = precision_score(array_la, array_pr, average='binary')
            r = recall_score(array_la, array_pr, average='binary')
            f1score = f1_score(array_la, array_pr, average='binary')
            IOU = iou(array_la, array_pr)

            acc_sum += acc.item()
            p_sum += p.item()
            r_sum += r.item()
            f1_sum += f1score.item()
            IOU_sum += IOU.item()

            num += 1

            # print('准确率：', acc)
            # print('召回率：', tpr)
            # print('精准率：',pre)
            # print('iou：',IOU)
            # print('f1:',f1)
            # print(num)

        print('acc_sum:', acc_sum)
        print('pre_sum:', p_sum)
        print('tpr_sum:', r_sum)
        print('f1score_sum:', f1_sum)
        print('IOU_sum:', IOU_sum)

        # print(len(val_loader))
        # print(len(test_loader))
        print("acc:{}, r:{},p:{},IOU:{},f1score:{}".format(acc_sum / len(val_loader),
                                                           r_sum / len(val_loader),
                                                           p_sum / len(val_loader),
                                                           IOU_sum / len(val_loader),
                                                           f1_sum / len(val_loader)
                                                           ))
        scheduler.step()


        print("epoch:{},train_loss:{}, val_loss:{}".format(epoch, train_loss / len(train_loader),
                                                           val_loss_512 / len(val_loader)))
        print('#################################')

        s2.append(val_loss_512 / len(val_loader))
        s8.append(val_loss_64 / len(val_loader))
        s9.append(val_loss_256 / len(val_loader))
        s10.append(val_loss_sum / len(val_loader))
        s3.append(acc_sum / len(val_loader))
        s4.append(r_sum / len(val_loader))
        s5.append(p_sum / len(val_loader))
        s6.append(IOU_sum / len(val_loader))
        s7.append(f1_sum / len(val_loader))

        y2 = s2
        y3 = s3
        y4 = s4
        y5 = s5
        y6 = s6
        y7 = s7
        y8 = s8
        y9 = s9
        y10 = s10

        y2 = pd.DataFrame(y2)
        y3 = pd.DataFrame(y3)
        y4 = pd.DataFrame(y4)
        y5 = pd.DataFrame(y5)
        y6 = pd.DataFrame(y6)
        y7 = pd.DataFrame(y7)
        y8 = pd.DataFrame(y8)
        y9 = pd.DataFrame(y9)
        y10 = pd.DataFrame(y10)
        w = pd.ExcelWriter('val_loss_GMRR26_1.xlsx')  # 写入Excel文件
        y2.to_excel(w, 'val_loss_512', float_format='%.16f')
        y8.to_excel(w, 'val_loss_64', float_format='%.16f')
        y9.to_excel(w, 'val_loss_256', float_format='%.16f')
        y10.to_excel(w, 'val_loss_sum', float_format='%.16f')
        y3.to_excel(w, 'acc', float_format='%.16f')
        y4.to_excel(w, 'tpr', float_format='%.16f')
        y5.to_excel(w, 'pre', float_format='%.16f')
        y6.to_excel(w, 'IOU', float_format='%.16f')
        y7.to_excel(w, 'f1score', float_format='%.16f')
        w.save()
        w.close()

def test():
    acc_sum = 0
    p_sum = 0
    IOU_sum = 0
    f1score_sum = 0
    r_sum=0
    num=0
    timess=0

    model= torch.load(model_path)
    # print('model:',model)

    for iter, (a,x,y) in enumerate(test_loader):
        # print(len(val_loader)) #170
        x_val, y_true_val = Variable(x), Variable(y)

        inputs_val = x_val.cuda()
        # labels_val = y_true_val.cuda()
#        print(inputs_val)
        start = time.perf_counter()
        # outputs_val = model(inputs_val)
        outputs_val = model(inputs_val)
        end = time.perf_counter()
        times = end - start
        # print("执行时间", times)
        timess += times
        # print("执行时间", end - start)
        # print("执行时间", timess)

        outputs_val = outputs_val.data.cpu().numpy()
        outputs_val = np.squeeze(outputs_val)
        outputs_val = np.transpose(outputs_val,(1,2,0))
        # a=visualize_feature_map(outputs_val)
        # print(a)
        pr = outputs_val.reshape((512,512, 2)).argmax(axis=2)
        array_pr = np.reshape(pr, -1)
        # print(array_pr)
        pr = pr.astype('uint8')
        # print(pr) #0 1
        pr[pr == 1] = 255
        # print(pr) #0 255
        skimage.io.imsave(os.path.join("E:/sunyan_yansan/2/luhui_code/dataset/ann_w_model", "%s_predict.png" % a), pr)

#评价指标
        lab = y_true_val.data.cpu().numpy()
        # print(lab)
        # print(lab.shape) #(1, 2, 512, 512)
        lab = np.squeeze(lab)
        # print(lab)
        # print(lab.shape) #(2, 512, 512)
        lab = np.transpose(lab, (1, 2, 0))
        # print(lab)
        # print(lab.shape) #(512, 512, 2)
        la = lab.reshape((512, 512, 2)).argmax(axis=2)
        # print(la)
        # print(la.shape) #(512, 512)
        array_la= np.reshape(la, -1)
        # print(array_la)
        # print(array_la.shape)
        # print(array_la.dtype) #int64

        acc=pixel_acc(array_la,array_pr)

        p=precision_score(array_la,array_pr , average='binary')
        r=recall_score(array_la,array_pr , average='binary')
        f1 = f1_score(array_la, array_pr, average='binary')

        IOU = iou(array_la, array_pr)

        acc_sum += acc.item()
        p_sum += p.item()
        r_sum += r.item()
        f1score_sum += f1.item()
        IOU_sum += IOU.item()


        num+=1

        # print('准确率：', acc)
        # print('召回率：', tpr)
        # print('精准率：',pre)
        # print('iou：',IOU)
        # print('f1:',f1)
        print(num)

    print('acc_sum:',acc_sum)
    print('pre_sum:', p_sum)
    print('tpr_sum:',r_sum)
    print('f1score_sum:', f1score_sum)
    print('IOU_sum:',IOU_sum)

    print(len(val_loader))
    print(len(test_loader))
    print("acc:{}, recall:{},pre:{},IOU:{},f1score:{}".format(acc_sum / len(test_loader),r_sum / len(test_loader),p_sum / len(test_loader),
                                                                        IOU_sum / len(test_loader),f1score_sum / len(test_loader)))
    print("执行时间", timess)



def iou(pred, target):

   ious = []

   for cls in range(n_class):
       pred_inds = pred == cls
       target_inds = target == cls
       intersection = pred_inds[target_inds].sum()
       union = pred_inds.sum() + target_inds.sum() - intersection
       if union == 0:
           ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
       else:
           ious.append(float(intersection) / max(union, 1))
       # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
   # return ious
   return ious[0]

def pixel_acc(pred, target):
   correct = (pred == target).sum()
   total   = (target == target).sum()
   return correct / total

#准确率
def get_acc(y, y_hat):
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)

#召回率
def get_tpr(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_positive / actual_positive

#精准率
def get_precision(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    predicted_positive = sum(y_hat)
    return true_positive / predicted_positive

#f1
def get_f1(x,y):  #x和y是召回率和精准率
    f=(2*x*y)/(x+y)
    return f

#特异度更多地被用于ROC曲线的绘制
def get_tnr(y, y_hat):
    true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
    actual_negative = len(y) - sum(y)
    return true_negative / actual_negative

#ROC曲线
def get_roc(y, y_hat_prob):
    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
        ret.append([get_tpr(y, y_hat), 1 - get_tnr(y, y_hat)])
    return ret


if __name__ == "__main__":

    train()
    # test()
