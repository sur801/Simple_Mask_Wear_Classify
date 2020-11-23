# -*- coding: utf-8 -*-
# /usr/bin/python3
import cv2
import numpy as np
import sys
import tensorflow as tf
from model import mask_convnet


model_path = './models'

mask_x = tf.placeholder(tf.float32, shape = [None, 2500])

y_conv = mask_convnet(mask_x)

probs = tf.nn.softmax(y_conv)

#model save
saver = tf.train.Saver()

# load check point
ckpt = tf.train.get_checkpoint_state(model_path)
sess = tf.Session()

# check point 불러옴.

data = []

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restore model Done")
tf.train.write_graph(sess.graph, "./", "train.pbtxt", as_text=True)
# save pbtxt file through 1 iteration with model

test_image = cv2.imread('sample/nomask3.jpg', cv2.IMREAD_GRAYSCALE)

input_image = cv2.resize(test_image, (50, 50))

data.append(input_image)
data = np.array(data).reshape(-1, 2500) * 1/255.0
#print(data.shape) # (1, 50, 50, 1)
result = sess.run(probs, feed_dict={mask_x : data})
if result is not None:
  print(result)
  max_result = result.max() # result = [[ 확률, 확률 ]] softmax 한 결과
  val=0
  for i in range(2):
    if result[0,i] == max_result:
      print(i) # 0 no mask, 1 mask