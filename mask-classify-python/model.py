#!/usr/bin/env python
# coding: utf-8
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
batch_pointer = 0 # pointer for fetch mini batch data
img_size=50 # input w, h size
train_data=[] # train image data
test_data=[] # test image
train_x=[] # x data for train
train_y=[] # y data for train
test_x=[] # x data for test
test_y=[] # y data for test
train_size, test_size=5000,749 # lenth of train, and test images
mini_batch_size = 10
path='/input/Medicalmask/Medicalmask/MedicalMask/images/'

# load data and split it to train, test data
def load_data():
    global train, train_data, test_data
    a=os.listdir(images)
    b=os.listdir(annotations)
    a.sort()
    b.sort()

    options=['face_with_mask','face_no_mask']
    train= train[train['classname'].isin(options)]
    train.sort_values('name',axis=0,inplace=True)

    bbox=[]
    

    # make bbox for train and test images
    for i in range(train_size+test_size):
        arr=[]
        for j in train.iloc[i][["x1",'x2','y1','y2']]:
            arr.append(j)
        bbox.append(arr)
    
    train["bbox"]=bbox  

    # use 4000 images for train
    for i in range(train_size):
        arr=[]
        for j in train.iloc[i]:
                arr.append(j)
        img_array=cv2.imread(os.path.join(images,arr[0]),0)
        #print(len(img_array))
        crop_image = img_array[arr[2]:arr[4],arr[1]:arr[3]]
        new_img_array=cv2.resize(crop_image,(img_size,img_size))
        train_data.append([new_img_array,arr[5]])

    for i in range(train_size,train_size+test_size):
        arr=[]
        for j in train.iloc[i]:
                arr.append(j)
        img_array=cv2.imread(os.path.join(images,arr[0]),0)
        #print(len(img_array))
        crop_image = img_array[arr[2]:arr[4],arr[1]:arr[3]]
        new_img_array=cv2.resize(crop_image,(img_size,img_size))
        #cv2.imwrite("test_images/"+str(i)+".jpg", crop_image)
        test_data.append([new_img_array,arr[5]])



def reshape_data(x, y, data):
    for features, labels in data:
        x.append(features)
        y.append(labels)
    lbl=LabelEncoder()
    y=lbl.fit_transform(y)

    x=np.array(x).reshape(-1,2500)
    x = x * 1 / 255.0 # normailze 0~1 
    x = np.float32(x) # make float32

    y = tf.keras.utils.to_categorical(y)

    y = np.array(y).reshape(-1,2)
    return x, y


def load_batch(size) :
    global batch_pointer
    x_batch = []
    y_batch = []
    for i in range(size):
        x_batch = train_x[batch_pointer:batch_pointer+size]
        y_batch = train_y[batch_pointer:batch_pointer+size]
    batch_pointer += size
    return x_batch, y_batch


def weight_variables(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride, padding='VALID'):
  return tf.nn.conv2d(x, W, strides=stride, padding=padding)

# at default, ksize = strides
def maxpool(x, padding='VALID'):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1] , padding = padding, strides=[1, 2, 2, 1])

def mask_convnet(x):
    # input shape (50, 50, 1)
    x_input = tf.reshape(x, [-1,50, 50, 1])

    #1 layer
    W_conv1 = weight_variables([3,3,1,100]) # w, h, input channel, output channel
    b_conv1 =  bias_variable([100])
    res_conv1 = conv2d(x_input, W_conv1, [1,2,2,1]) + b_conv1 # Conv 연산
    scale1 = tf.ones(shape=[100])
    offset1 = tf.zeros(shape=[100])
    batch_norm1, _, _ = tf.nn.fused_batch_norm(res_conv1, scale1, offset1, is_training = True) # 결과 값 batch norm 된 결과만 받고, mean , variance 무시
    acti_relu1 = tf.nn.relu(batch_norm1)
    Max1 = maxpool(acti_relu1)
    
    #2 layer
    W_conv2 = weight_variables([3, 3, 100, 64])
    b_conv2 = bias_variable([64])
    res_conv2 = conv2d(Max1, W_conv2, [1,1,1,1]) + b_conv2 # Conv 연산
    scale2 = tf.ones(shape=[64])
    offset2 = tf.zeros(shape=[64])
    batch_norm2, _, _ = tf.nn.fused_batch_norm(res_conv2, scale2, offset2, is_training = True) 
    activ_relu2 = tf.nn.relu(batch_norm2)
    Max2 = maxpool(activ_relu2)


    #3 layer fully connected layer
    W_fc1 = weight_variables([5*5*64,50] )
    b_fc1 = bias_variable([50])
    Flat = tf.reshape(Max2, [-1, 5*5*64]) # flatten
    Fc1 = tf.nn.relu(tf.matmul(Flat, W_fc1) + b_fc1)

    #4 layer fully connected layer
    W_fc2 = weight_variables([50,2] )
    b_fc2 = bias_variable([2])
    Fc2 = tf.add(tf.matmul(Fc1, W_fc2) ,b_fc2)
    return Fc2


#shuffle train data
def suffle_data():
    global train_x, train_y
    data_set = list(zip(train_x,train_y))
    random.shuffle(data_set)
    random.shuffle(data_set) # shuffle twice
    train_x, train_y = zip(*data_set)


def train_model() :
    global batch_pointer
    train_epoch = 100
    x = tf.placeholder(tf.float32, shape = [None, 2500])
    y_ =tf.placeholder(tf.float32, shape = [None, 2])

    y_out = mask_convnet(x)    #(이미지 개수, 2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
    
    # beta1 for momentumn decay
    # lr = 1e-3 => 70
    # lr = 1e-2 -> 73
    train_step = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # use adam optimizer
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for epoch in range(train_epoch+1):
            # 전체 이미지 수 / batch size
            # mini batch 개수를 구해서 그만큼 반복
            for b in range(int(train_size/mini_batch_size)): #5000/100 = 50번 실행
                x_mini_batch , y_mini_batch = load_batch(mini_batch_size) 
                #if b % 20 == 0 : 
                #    # sess.run 과 같음 근데 이건 accuracy 값이 나오는거.
                #    train_accuracy = accuracy.eval(feed_dict={x: x_mini_batch, y_: y_mini_batch})
                #    print("step : %d, training accuracy : %g"% (b, train_accuracy))

                train_step.run(feed_dict={x: x_mini_batch, y_: y_mini_batch}) # optimizer run

            batch_pointer = 0
            suffle_data() # 1개의 epoch 끝났으니까 데이터 섞어서 다시
            if epoch == train_epoch :
                saver.save(sess, './models/mask_model', global_step = epoch)

        # finish training
        print("Test Accuracy: ",accuracy.eval(session=sess, feed_dict={x: test_x, y_: test_y}))


def main():
    global train_x, train_y, test_x, test_y
    #with tf.device('/device:CPU:0'): # cpu로 돌릴때 이렇게
    load_data() # load data from path
    train_x, train_y = reshape_data(train_x, train_y, train_data) # preprocess image data
    test_x, test_y = reshape_data(test_x, test_y, test_data) 
    train_model() # train images with convnet.

if __name__ == "__main__":
    main()
