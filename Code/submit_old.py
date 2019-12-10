# model
import pandas as pd
from keras.models import load_model
from keras.layers import Lambda
import keras.backend as K
from keras.models import Sequential, Model
import tensorflow as tf
import cv2
import numpy as np

def resize_image(img, input_width = 512, input_height = 512):
    img = cv2.resize(img, (input_width, input_height))
    return (img / 255).astype('float32')

def CreateMaskImages(imageName):

    trainimage = cv2.imread(PATH  + "/train_images/" + imageName + '.jpg')
    imagemask = cv2.imread(PATH + "/train_masks/" + imageName + ".jpg",0)
    try:
        imagemaskinv = cv2.bitwise_not(imagemask)
        res = cv2.bitwise_and(trainimage,trainimage,mask = imagemaskinv)
        return res
    except:
        return trainimage


PATH = './Data/'
test = pd.read_csv(PATH + 'sample_submission.csv')
#model = HourglassNetwork(heads=heads, **kwargs)
model = load_model('./centernet.h5')
def CreateMaskImages(imageName):

    trainimage = cv2.imread(PATH  + "/test_images/" + imageName + '.jpg')
    imagemask = cv2.imread(PATH + "/test_masks/" + imageName + ".jpg",0)
    try:
        imagemaskinv = cv2.bitwise_not(imagemask)
        res = cv2.bitwise_and(trainimage,trainimage,mask = imagemaskinv)
        return res
    except:
        return trainimage

def str_to_coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def coords_to_str(coords):
    s = []
    for c in coords:
        for n in range(7):
            s.append(str(c[n]))
    return ' '.join(s)


def test_generator(test, batch_size=4):
    while True:
        for i in range(len(test)):
            img = CreateMaskImages(test['ImageId'][i])
            img = resize_image(img)
            X_batch = img[np.newaxis, :]

            yield (X_batch)

#model = CtDetDecode(model)
def extract_coords(hm, reg, flipped=False):
    logits = hm
    regr_output = reg
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


for i in range(len(test)):
    img = CreateMaskImages(test['ImageId'][i])
    img = resize_image(img)
    X_batch = img[np.newaxis, :]

    reg_head, hm_head = model.predict(X_batch)
    hm_head = hm_head.reshape(128, 128,1)
    reg_head = reg_head.reshape(128,128,6)
    print()

def unused():
    submission = []
    detections = []
    for d in detections:

        yaw, pitch, roll, x, y, z, score = d
        if score < 0.3:
            continue
        else:
            submission.append(d)

    Prediction_string = coords_to_str(submission)
    test['PredictionString'][i] = Prediction_string

#test.to_csv('submission.csv', index=True)