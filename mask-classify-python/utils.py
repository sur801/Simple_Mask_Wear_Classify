import numpy as np 
import pandas as pd 
import os
import sys
from utils import *
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import tensorflow as tf
import random
from sklearn.preprocessing import LabelEncoder



images=os.path.join("input/Medicalmask/Medicalmask/MedicalMask/images")
annotations=os.path.join("input/Medicalmask/Medicalmask/MedicalMask/annotations")
train=pd.read_csv(os.path.join("input/train.csv"))
submission=pd.read_csv(os.path.join("input/submission.csv"))
batch_pointer = 0
img_size=50
data=[]
train_images = []
test_images = []
train_x=[]
train_y=[]

path='/input/Medicalmask/Medicalmask/MedicalMask/images/'

def load_data():
    global train, train_images, test_images, data
    a=os.listdir(images)
    b=os.listdir(annotations)
    a.sort()
    b.sort()

    train_images=a[1698:]
    test_images=a[:1698]

    options=['face_with_mask','face_no_mask']
    train= train[train['classname'].isin(options)]
    train.sort_values('name',axis=0,inplace=True)

    bbox=[]

    for i in range(len(train)):
        arr=[]
        for j in train.iloc[i][["x1",'x2','y1','y2']]:
            arr.append(j)
        bbox.append(arr)
    train["bbox"]=bbox  

    for i in range(len(train)):
        arr=[]
        for j in train.iloc[i]:
                arr.append(j)
        img_array=cv2.imread(os.path.join(images,arr[0]),0)
        #print(len(img_array))
        crop_image = img_array[arr[2]:arr[4],arr[1]:arr[3]]
        new_img_array=cv2.resize(crop_image,(img_size,img_size))
        data.append([new_img_array,arr[5]])
    print(len(data))




def reshape_data():
    global train_x,train_y
    for features, labels in data:
        train_x.append(features)
        train_y.append(labels)
    lbl=LabelEncoder()
    train_y=lbl.fit_transform(train_y)

    train_x=np.array(train_x).reshape(-1,2500)
    print(train_x.shape)
    print(train_y.shape)
    train_x = train_x * 1 / 255.0 # normailze 0~1 
    train_x = np.float32(train_x) # make float32

    train_y = tf.keras.utils.to_categorical(train_y)

    train_y = np.array(train_y).reshape(-1,2)