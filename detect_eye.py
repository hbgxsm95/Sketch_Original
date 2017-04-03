
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

PI = 3.1415926
EYE_DISTANCE = 50
EYE_EDGE_DISTANCE_X = 75
EYE_EDGE_DISTANCE_Y = 125
CROP_WIDTH = 200
CROP_HEIGHT = 250
import sys
caffe_root = './'
sys.path.insert(0, caffe_root + 'python')
sys.path.append("/Users/zhaotsuchikaqin/caffe/python")
import caffe
import os

def point_rot(point, affineMat):

    #    # point.shape, affineMat.shape, affineMat[0,2], point[1]
        res = np.array([-1,-1])
        res[0] = affineMat[0,0]*point[1] + affineMat[0,1]*point[0] + affineMat[0,2]
            
        res[1] = affineMat[1,0]*point[1] + affineMat[1,1]*point[0] + affineMat[1,2]

        return res

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    #plt.imshow(data); plt.axis('off')

def pre_process(img, lefteye):
    
    crop_x = lefteye[1] - 50 if (lefteye[1] - 50 > 0) else 0
    crop_y = lefteye[0] - 50 if (lefteye[0] - 50 > 0) else 0
    img = img[crop_y:crop_y + 150, crop_x:crop_x + 150];
    return img, crop_x, crop_y


def get_netoutput(im_orig,net):
    
    im = cv2.resize(im_orig, (32,32))
    #cv2.imwrite('init.png', im)


    # im.shape
    im = im.astype(np.float32, copy=True)

    input_blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    input_blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
 
    channel_swap = (0, 3, 1, 2)
    input_blob = input_blob.transpose(channel_swap)
    net.blobs['data'].reshape(*(input_blob.shape))

    net.blobs['data'].data[...] = input_blob

    net.forward()
    map_1 = net.blobs['heat-map-pred'].data[0][0]
    map_2 = net.blobs['heat-map-pred'].data[0][1]    


    
    x = zip(*np.unravel_index(map_1.argsort(axis=None), dims=map_1.shape))[::-1][:10]
    array = np.array(x)
    # left eye coordinate, you need to change it into pre img, cur img size is 40*40
    # np.mean(array, axis=0) 
    lefteye = np.mean(array, axis=0)


    lefteye[0] *= im_orig.shape[0] / 32.0
    lefteye[1] *= im_orig.shape[1] / 32.0

    y = zip(*np.unravel_index(map_2.argsort(axis=None), dims=map_2.shape))[::-1][:10]
    array = np.array(y)
    # right eye coordinate, you need to change it into pre img, cur img size is 40*40
    # np.mean(array, axis=0)
    righteye= np.mean(array, axis=0)

    righteye[0] *= im_orig.shape[0] / 32.0
    righteye[1] *= im_orig.shape[1] / 32.0

    return lefteye, righteye


def caffeDetect(filename):
    if os.path.isfile(caffe_root + 'model/eyeDetector/conf5_iter_162000.caffemodel'):   
        print 'Model found.'
    else:
        print 'Model Not Exist!.'

    caffe.set_mode_cpu()

    model_def = caffe_root + 'model/eyeDetector/deploy_output_resize_map.prototxt'
    model_weights = caffe_root + 'model/eyeDetector/conf5_iter_162000.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    # get img
    im_orig = cv2.imread(filename)
    # im_orig.shape
    img = im_orig




    lefteye, righteye = get_netoutput(im_orig,net)
    print lefteye, righteye

    #cv2.imshow('init.png', img)
    #cv2.waitKey(1000)


    # we first estimate the eye then crop the area for finetune

    estimate_img, crop_x, crop_y = pre_process(im_orig, lefteye)

    ## estimate_img.shape
    #cv2.imshow('pre_process.png',estimate_img)
    #cv2.waitKey(1000)

    lefteye, righteye = get_netoutput(estimate_img,net)

    lefteye[1] += crop_x
    righteye[1] += crop_x

    lefteye[0] += crop_y
    righteye[0] += crop_y


    #cv2.circle(img, (int(lefteye[1]), int(lefteye[0])),  1, (0,255,0))
    #cv2.circle(img, (int(righteye[0]), int(righteye[1])),  1, (0,255,0))

    #cv2.imshow('finetune.png', img)
    #cv2.waitKey(1000)
    #cv2.imwrite('finetune.png', img)


    dis_x = righteye[1] - lefteye[1]
    dis_y = righteye[0] - lefteye[0]

    if dis_x == 0:
       angle = -90 if (lefteye['y'] > righteye['y']) else 90
    else:
       angle = math.atan(dis_y/dis_x)*180/PI


    # 'angle is ', angle
    center = (img.shape[1]/2, img.shape[0]/2)

    affineMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    ## affineMat, affineMat.shape
    rotate_img = cv2.warpAffine(img, affineMat, (img.shape[1], img.shape[0]))

    lefteye = point_rot(lefteye, affineMat)

    righteye = point_rot(righteye, affineMat)

    # 'after rotate'
    # lefteye, righteye
    #cv2.circle(rotate_img, (int(lefteye[0]), int(lefteye[1])),  1, (0,0,255))
    #cv2.circle(rotate_img, (int(righteye[0]), int(righteye[1])),  1, (0,0,255))
    #cv2.imshow('rotate_img', rotate_img)
    #cv2.waitKey(1000)
    #cv2.imwrite('rotata_img.png', rotate_img)

    dis_x = righteye[0] - lefteye[0];
    dis_y = righteye[1] - lefteye[1];

    distance = math.sqrt(dis_x * dis_x + dis_y * dis_y)
    # 'eye distance is ', distance 

    scale = EYE_DISTANCE/distance

    # 'scale is ', scale
    img = cv2.resize(rotate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)




    # lefteye, righteye
    lefteye = lefteye * scale
    righteye = righteye * scale

    # 'left eye is ', lefteye
    # cv2.circle(img, (int(lefteye[0]), int(lefteye[1])),  1, (0,255,0))

    #cv2.imshow('scale_img', img)
    #cv2.waitKey(1000)


    #cv2.imwrite('scale.png', img)


    min_x = int(lefteye[0] - EYE_EDGE_DISTANCE_X)
    min_y = int(lefteye[1] - EYE_EDGE_DISTANCE_Y)

    min_x = 0 if(min_x < 0) else min_x
    min_y = 0 if(min_y < 0) else min_y


    max_x = min_x + CROP_WIDTH if(min_x + CROP_WIDTH < img.shape[1]) else img.shape[1]
    max_y = min_y + CROP_HEIGHT if(min_y + CROP_HEIGHT < img.shape[0]) else img.shape[0]

    # img.shape
    img = img[min_y:max_y, min_x:max_x]

    # img.shape, min_x, max_x

    ret = np.ones((1000, 500, 3), dtype = np.uint8) * 255

    start_x = 0
    start_y = 0
    if lefteye[0] - EYE_EDGE_DISTANCE_X < 0:
       start_x = EYE_EDGE_DISTANCE_X - lefteye[0]
    if lefteye[1] - EYE_EDGE_DISTANCE_Y < 0:
       start_y = EYE_EDGE_DISTANCE_Y - lefteye[1]


    # start_y, start_y + max_y - min_y, start_x, start_x + max_x - min_x, img.shape
    ret[start_y:start_y + max_y - min_y, start_x:start_x + max_x - min_x,:] = img;


    ret = ret[0:250, 0:200];

    #cv2.imshow('out_img', ret)
    #cv2.waitKey(1000)


    #cv2.imwrite('final.png', ret)
    #res = Image.fromarray(ret)
    #res.show()
    return ret

    
    #plt.show()

