import numpy as np
#from libtiff import TIFF
import os
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import random
import cv2
from functools import partial
import numpy as np
import tensorflow as tf
from skimage.measure import compare_ssim as ssim
import keras
from keras.layers import Layer, InputSpec, Reshape
from keras.layers import Input, Add, Concatenate, Lambda
from keras.layers import LeakyReLU
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, Dropout
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Model,load_model
import keras.backend as K
from keras.initializers import RandomNormal
from numpy import load
from sklearn.metrics import confusion_matrix,jaccard_similarity_score,f1_score,roc_auc_score,auc,recall_score, auc,roc_curve
import gc
import glob
import pycm
import warnings
warnings.filterwarn


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        if type(padding) == int:
            padding = (padding, padding)
        self.padding = padding
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)
    def get_config(self):
      cfg = super().get_config()
      return cfg   

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


def normalize_pred(img):
    img = np.reshape(img,[1,64,64,1])
    img_coarse = tf.image.resize(img, (32,32), method=tf.image.ResizeMethod.LANCZOS3)
    img_coarse = (img_coarse - 127.5) / 127.5
    img_coarse = np.array(img_coarse)
    
    X_fakeB_coarse,x_global = g_global_model.predict(img_coarse)
    X_fakeB_coarse = (X_fakeB_coarse+1)/2.0
    #X_fakeB_coarse = ((X_fakeB_coarse + 1) * 127.5).astype('uint8')
    #X_fakeB_coarse = cv2.normalize(X_fakeB_coarse, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    pred_img_coarse = X_fakeB_coarse[:,:,:,0]


    img = (img - 127.5) / 127.5
    X_fakeB = g_local_model.predict([img,x_global])
    X_fakeB = (X_fakeB+1)/2.0
    #X_fakeB = ((X_fakeB + 1) * 127.5).astype('uint8')
    #X_fakeB = cv2.normalize(X_fakeB, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    pred_img = X_fakeB[:,:,:,0]
    return [np.asarray(pred_img,dtype=np.float32),np.asarray(pred_img_coarse,dtype=np.float32)]

def strided_crop(img, img_h,img_w,height, width,stride=1):

    full_prob = np.zeros((img_h, img_w),dtype=np.float32)
    full_sum = np.ones((img_h, img_w),dtype=np.float32)
    
    max_x = int(((img.shape[0]-height)/stride)+1)
    #print("max_x:",max_x)
    max_y = int(((img.shape[1]-width)/stride)+1)
    #print("max_y:",max_y)
    max_crops = (max_x)*(max_y)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
                crop_img_arr = img[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                [pred,pred_256] = normalize_pred(crop_img_arr)
                full_prob[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += pred[0]
                full_sum[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += 1
                i = i + 1
                #print(i)
    out_img = full_prob / full_sum
    return out_img