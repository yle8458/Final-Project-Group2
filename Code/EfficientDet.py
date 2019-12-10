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

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train['PredictionString']:
        coords = str2coords(ps)
        arr += [c[col] for c in coords]
    points_df[col] = arr

print('len(points_df)', len(points_df))

def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x

def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

from math import sin, cos

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image

IMG_WIDTH, IMG_HEIGHT = 2048, 640
MODEL_SCALE = 8


def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict


def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)

    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict




def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg_v = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg_l = bg_v[:, :img.shape[1] // 8]
    bg_r = bg_v[:, :int(img.shape[0] * 3.2) - img.shape[1] // 8 - img.shape[1]]

    img = np.concatenate([bg_l, img, bg_r], 1)

    # img = img[img.shape[0] // 2:]
    # bg = bg[:img.shape[1] - img.shape[0], :]
    # img = np.concatenate([img, bg_down], 0)
    #print(img.shape)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:, ::-1]
    return (img / 255).astype('float32')


def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x_scale, y_scale = y, x
        x_scale = (x_scale - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x_scale = np.round(x_scale).astype('int')
        y_scale = (y_scale + img.shape[1] // 8) * IMG_WIDTH / (img.shape[0] * 16 / 5) / MODEL_SCALE
        y_scale = np.round(y_scale).astype('int')
        if x_scale >= 0 and x_scale < IMG_HEIGHT // MODEL_SCALE and y_scale >= 0 and y_scale < IMG_WIDTH // MODEL_SCALE:
            mask[x_scale, y_scale] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x_scale, y_scale] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:, ::-1])
        regr = np.array(regr[:, ::-1])
    return mask, regr


DISTANCE_THRESH_CLEAR = 2


def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    # borrowed from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

zy_slope = LinearRegression()
X = points_df[['z']]
y = points_df['y']
zy_slope.fit(X, y)

xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)

def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx, z]])[0] - y) ** 2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 8) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x - r) ** 2 + (y - c) ** 2) + max(0.4, slope_err)

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new


def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]


def extract_coords(prediction, flipped=False):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > 0)
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
            optimize_xy(r, c,
                        coords[-1]['x'],
                        coords[-1]['y'],
                        coords[-1]['z'], flipped)
    coords = clear_duplicates(coords)
    return coords


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)


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


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


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

predictions = []

test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)

model.eval()

for img, _, _ in tqdm(test_loader):
    with torch.no_grad():
        output = model(img.to(device))
    output = output.data.cpu().numpy()
    for out in output:
        coords = extract_coords(out)
        s = coords2str(coords)
        predictions.append(s)
test = pd.read_csv(PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv('predictions.csv', index=False)
test.head()