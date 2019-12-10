import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm#_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
from scipy.optimize import minimize


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils

PATH = './Data/'
os.listdir(PATH)

IMG_SHAPE = [2710, 3384]
BATCH_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 20

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')

# From camera.zip
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1

        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr]


train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

df_train, df_val = train_test_split(train, test_size=0.01, random_state=42)
df_test = test

# Create dataset objects
train_dataset = CarDataset(df_train, train_images_dir, training=True)
val_dataset = CarDataset(df_val, train_images_dir, training=False)
test_dataset = CarDataset(df_test, test_images_dir, training=False)


# Create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

from efficientnet_pytorch import EfficientNet


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))


        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh
class myEfficientNet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(myEfficientNet, self).__init__(blocks_args=blocks_args, global_params=global_params)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        P = []
        index = 0
        num_repeat = 0
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            num_repeat = num_repeat + 1
            if (num_repeat == self._blocks_args[index].num_repeat):
                num_repeat = 0
                index = index + 1
                P.append(x)
        return P

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        # Convolution layers
        P = self.extract_features(inputs)
        return P

    def get_list_features(self):
        list_feature = []
        for idx in range(len(self._blocks_args)):
            list_feature.append(self._blocks_args[idx].output_filters)

        return list_feature


class BiFPN(nn.Module):
    def __init__(self, num_out = 64):
        super(BiFPN, self).__init__()

        self.num_out = num_out
        #weighted
        self.w1 = nn.Parameter(torch.Tensor(5, 3).fill_(0.5))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 3).fill_(0.5))
        self.relu2 = nn.ReLU()

        # start from layer 3


        self.conv4up = nn.Sequential(
            nn.Conv2d(self.num_out, self.num_out, kernel_size=3, stride=1, padding=1, groups=self.num_out),
            nn.BatchNorm2d(num_features=self.num_out), nn.ReLU())
        self.conv5up = nn.Sequential(
            nn.Conv2d(self.num_out, self.num_out, kernel_size=3, stride=1, padding=1, groups=self.num_out),
            nn.BatchNorm2d(num_features=self.num_out), nn.ReLU())
        self.conv6up = nn.Sequential(
            nn.Conv2d(self.num_out, self.num_out, kernel_size=3, stride=1, padding=1, groups=self.num_out),
            nn.BatchNorm2d(num_features=self.num_out), nn.ReLU())

        self.conv3dw = nn.Sequential(
            nn.Conv2d(self.num_out, self.num_out, kernel_size=3, stride=1, padding=1, groups=self.num_out),
            nn.BatchNorm2d(num_features=self.num_out), nn.ReLU())
        self.conv4dw = nn.Sequential(
            nn.Conv2d(self.num_out, self.num_out, kernel_size=3, stride=1, padding=1, groups=self.num_out),
            nn.BatchNorm2d(num_features=self.num_out), nn.ReLU())
        self.conv5dw = nn.Sequential(
            nn.Conv2d(self.num_out, self.num_out, kernel_size=3, stride=1, padding=1, groups=self.num_out),
            nn.BatchNorm2d(num_features=self.num_out), nn.ReLU())
        self.conv6dw = nn.Sequential(
            nn.Conv2d(self.num_out, self.num_out, kernel_size=3, stride=1, padding=1, groups=self.num_out),
            nn.BatchNorm2d(num_features=self.num_out), nn.ReLU())
        self.conv7dw = nn.Sequential(
            nn.Conv2d(self.num_out, self.num_out, kernel_size=3, stride=1, padding=1, groups=self.num_out),
            nn.BatchNorm2d(num_features=self.num_out), nn.ReLU())



    def forward(self, inputs):
        num_channels = self.num_out
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + 0.0001  # normalize
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + 0.0001
        # upsample network

        scale = (P6_in.shape[3] / P7_in.shape[3])
        P6_up = self.conv6up(w1[0, 0] * P6_in + w1[0, 1] * self.Resize(scale_factor=scale)(P7_in))
        scale = (P5_in.shape[3] / P6_up.shape[3])
        P5_up = self.conv5up(w1[1, 0] * P5_in + w1[1, 1] * self.Resize(scale_factor=scale)(P6_up))
        scale = (P4_in.shape[3] / P5_up.shape[3])
        P4_up = self.conv4up(w1[2, 0] * P4_in + w1[2, 1] * self.Resize(scale_factor=scale)(P5_up))
        scale = (P3_in.shape[3] / P4_up.shape[3])
        P3_out = self.conv3dw(w1[3,0] * P3_in + w1[3, 1] * self.Resize(scale_factor=scale)(P4_up))

        # fix to downsample by interpolation
        # downsample networks
        P4_out = self.conv4dw(w2[0, 0] * P4_in + w2[0, 1] * P4_up + w2[0,2] * F.interpolate(P3_out, P4_up.size()[2:]))
        P5_out = self.conv5dw(w2[1, 0] * P5_in + w2[1, 1] * P5_up + w2[1,2] * F.interpolate(P4_out, P5_up.size()[2:]))
        P6_out = self.conv6dw(w2[2, 0] * P6_in + w2[2, 1] * P6_up + w2[2,2] * F.interpolate(P5_out, P6_up.size()[2:]))
        P7_out = self.conv7dw(w1[4, 0] * P7_in + w1[4, 1] * F.interpolate(P6_out, P7_in.size()[2:]))
        return (P3_out, P4_out, P5_out, P6_out, P7_out)

    @staticmethod
    def Conv(in_channels, out_channels, kernel_size, stride, padding, groups=1):
        features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        return features

    @staticmethod
    def Resize(scale_factor=2, mode='bilinear'):
        upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsample


class RegressionModel(nn.Module):
    def __init__(self, num_features_in = 64, feature_size=64):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.out = nn.Conv2d(feature_size, 7, kernel_size = 1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)

        out = self.out(out)

        #out = out.permute(0, 2, 3, 1)
        return out


class ClassificationModel(nn.Module):
    def __init__(self, phi = 0, num_anchors=9, num_classes=1, prior=0.01, feature_size=64):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        num_class_layers = 3 + phi // 3

        modules = []

        for i in range(num_class_layers):
            modules.append(nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1))
            modules.append(nn.ReLU())

        self.net = nn.Sequential(*modules)

        self.output = nn.Conv2d(feature_size, num_classes, kernel_size=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.net(x)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        #out1 = out.permute(0, 2, 3, 1)

        #batch_size, width, height, channels = out1.shape

        #out2 = out1.view(batch_size, width, height,  self.num_classes)

        return out


class MyEfficientDet(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes):
        super(MyEfficientDet, self).__init__()
        self.base_model = myEfficientNet.from_pretrained('efficientnet-b0')

        # bifpn with 64 channels
        self.bifpn1 = BiFPN(64)
        self.bifpn2 = BiFPN(64)


        self.in_channels = self.base_model.get_list_features()
        self.conv3in = nn.Sequential(
            nn.Conv2d(self.in_channels[2], 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.conv4in = nn.Sequential(
            nn.Conv2d(self.in_channels[3], 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.conv5in = nn.Sequential(
            nn.Conv2d(self.in_channels[4], 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.conv6in = nn.Sequential(
            nn.Conv2d(self.in_channels[5], 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.conv7in = nn.Sequential(
            nn.Conv2d(self.in_channels[6], 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64), nn.ReLU())

        self.regression = RegressionModel()
        self.classification = ClassificationModel()

        self.outc = nn.Conv2d(8*5, n_classes, 1)
        self.freeze_bn()

    @staticmethod
    def Resize(scale_factor=2, mode='bilinear'):
        upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsample

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        batch_size = x.shape[0]

        feats = self.base_model.extract_features(x)
        P1, P2, P3, P4, P5, P6, P7 = feats

        P7 = self.conv7in(P7)
        P6 = self.conv6in(P6)
        P5 = self.conv5in(P5)
        P4 = self.conv4in(P4)
        P3 = self.conv3in(P3)

        x1 = self.bifpn1([P3, P4, P5, P6, P7])
        x2 = self.bifpn2(x1)
        regs = [self.regression(x) for x in x2]
        clss = [self.classification(x) for x in x2]

        # Aggregate the features
        for i in range(4, 0, -1):
            scale = regs[i - 1].shape[-1] / regs[i].shape[-1]
            regs[i - 1] = torch.cat([self.Resize(scale_factor=scale)(regs[i]) , regs[i - 1] ], dim=1)
            clss[i - 1] = torch.cat([self.Resize(scale_factor=scale)(clss[i]) , clss[i - 1] ], dim=1)

        xout = torch.cat([clss[0], regs[0]], dim=1)
        xout = self.outc(xout)

        return xout

model = MyEfficientDet(8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)


def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = mask_loss + regr_loss
    #loss = regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss


def train_model(epoch, history=None):
    model.train()

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)

        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

        loss.backward()

        optimizer.step()
        exp_lr_scheduler.step()

    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))


def evaluate_model(epoch, history=None):
    model.eval()
    loss = 0

    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss += criterion(output, mask_batch, regr_batch, size_average=False).data

    loss /= len(dev_loader.dataset)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(loss))


import gc

history = pd.DataFrame()

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_model(epoch, history)
    evaluate_model(epoch, history)

torch.save(model.state_dict(), './model.pth')
