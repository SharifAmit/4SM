import numpy as np
from model import fine_generator, coarse_generator
#from libtiff import TIFF
import os
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import random
import cv2
from functools import partial
import numpy as np
import tensorflow as tf
import keras
import argparse
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
warnings.filterwarnings('ignore')



def normalize_pred(img,g_global_model,g_local_model):
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

def strided_crop(img, img_h,img_w,height, width,g_global_model,g_local_model,stride=1):

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
                [pred,pred_256] = normalize_pred(crop_img_arr,g_global_model,g_local_model)
                full_prob[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += pred[0]
                full_sum[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += 1
                i = i + 1
                #print(i)
    out_img = full_prob / full_sum
    return out_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='test', help='path/to/save/dir')
    parser.add_argument('--weight_name', type=str, default='test', help='.h5 file name')    
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--savedir', type=str, required=False, help='path/to/save_directory',default='RVGAN')
    parser.add_argument('--resume_training', type=str, required=False,  default='no', choices=['yes','no'])
    parser.add_argument('--inner_weight', type=float, default=0.5)
    args = parser.parse_args()


    K.clear_session()
    gc.collect()

    stride = args.stride # Change Stride size to 8 or 16 for faster inference prediction
    crop_size_h = args.crop_size
    crop_size_w = args.crop_size
    weight_name = args.weight_name
    in_dir = args.in_dir
    directory = in_dir+'/pred'

    if not os.path.exists(directory):
        os.makedirs(directory)
    f = glob.glob(in_dir+"/JPEGImages/*.jpg")

    img_shape = (64,64,1)
    label_shape = (64,64,1)
    x_global = (32,32,64)
    opt = Adam()

    g_local_model = fine_generator(x_global,img_shape)
    g_local_model.load_weights('weight_file/local_model_'+weight_name+'.h5')
    g_local_model.compile(loss='mse', optimizer=opt)
    
    img_shape_g = (32,32,1)
    g_global_model = coarse_generator(img_shape_g,n_downsampling=2, n_blocks=9, n_channels=1)
    g_global_model.load_weights('weight_file/global_model_'+weight_name+'.h5')
    g_global_model.compile(loss='mse',optimizer=opt)

    for files in f:
        fo = files.split('\\')
        img = Image.open(files)
        img_arr = np.asarray(img)
        height, width, channel = img_arr.shape
        filename_with_ext = fo[1].split('.')
        filename = filename_with_ext[0]

        img_name = in_dir+"/JPEGImages/"+filename+".jpg"
        img = Image.open(img_name)
        img_arr = np.asarray(img,dtype=np.float32)
        img_arr = img_arr[:,:,0]
        out_img = strided_crop(img_arr, img_arr.shape[0], img_arr.shape[1], crop_size_h, crop_size_w,g_global_model,g_local_model,stride)
        out_img_sv = out_img.copy()
        out_img_sv = ((out_img_sv) * 255.0).astype('uint8')

        out_img_sv = out_img_sv.astype(np.uint8)
        out_im = Image.fromarray(out_img_sv)
        out_im_name = directory+'/'+fo[1]
        out_im.save(out_im_name)