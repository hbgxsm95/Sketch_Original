"""
This script handles the skimage exif problem.
"""


from PIL import Image
import numpy as np

import math
import json
import sys
import cv
import cv2
import detect_eye

sys.path.append("/Users/zhaotsuchikaqin/caffe/python")
sys.path.append("/usr/local/Cellar/opencv/2.4.12/lib/python2.7/site-packages")
import caffe


PI = 3.1415926
EYE_DISTANCE = 50
EYE_EDGE_DISTANCE_X = 75
EYE_EDGE_DISTANCE_Y = 125

CROP_WIDTH = 200
CROP_HEIGHT = 250

OUTPUT_HEIGHT = 238
OUTPUT_WIDTH = 188

PARSING_HEIGHT = 125
PARSING_WIDTH = 100

class Sketcher():

    caffe_net = caffe.Net("model/p2s/deploy_6layers.prototxt","model/p2s/model_iter_200000.caffemodel",1)
    #caffe_net.set_phase_test()
    #caffe_net.set_mode_gpu()
    #caffe_net_hair = caffe.Net("model/deploy_6layers_hair.prototxt", "model/model_iter_250000.caffemodel",1)
    #caffe_net_hair.set_phase_test()
    #caffe_net_hair.set_mode_gpu()
    #caffe_net_parsing = caffe.Net("model/deploy_6layers_parsing.prototxt", "model/model_iter_400000.caffemodel",1)
    #caffe_net_parsing.set_phase_test()
    #caffe_net_parsing.set_mode_gpu()

    #rotate center is (50,50), rotate angle in degrees conter-clockwise
    def point_rot(point, affineMat):

    #    print point.shape, affineMat.shape, affineMat[0,2], point[1]
        res = np.array([-1,-1])
        res[0] = affineMat[0,0]*point[1] + affineMat[0,1]*point[0] + affineMat[0,2]
            
        res[1] = affineMat[1,0]*point[1] + affineMat[1,1]*point[0] + affineMat[1,2]

        return res

    # crop to resized face image
    def crop(self, filename):
        img = Image.open(filename)
        
        #print('Facepp Recognize result:', eyeDetector.faceppDetect(filename))
        #print('Opencv Recognize result:', eyeDetector.opencvDetect(filename))
        #result = eyeDetector.opencvDetect(filename)
        result = detect_eye.caffeDetect(filename)
        res = Image.fromarray(result)
        #res.show()
        return res
        #return img
        

    def origin(self,filename):
        img = Image.open(filename)
        return img


    # transform photo to sketch image
    def process(self, img):
        scale = 0.003921569
        mean = 128

        mean_file = "model/p2s/sketch_mean.png"
        #face_file = "model/FFFFFFF_L.png"
        #back_file = "model/BACK.png"
        #hair_file = "model/HAIR.png"
    
        mean_img = Image.open(mean_file)
        #face_img = Image.open(face_file)
        #back_img = Image.open(back_file)
        #hair_img = Image.open(hair_file)

        #input process
        mean_in = np.array(mean_img.getdata(), dtype=np.float32)
        mean_in = mean_in.reshape(1, 1, CROP_HEIGHT, CROP_WIDTH)-mean
    
        img_in = np.array(img.getdata(), dtype=np.float32)
        img_in = np.transpose(img_in)
        img_in = img_in.reshape(1, 3, CROP_HEIGHT, CROP_WIDTH)- mean

        ##parsing_sub1 = np.array(back_img.getdata(), dtype=np.float32)
        ##parsing_sub2 = np.array(face_img.getdata(), dtype=np.float32)
        ##parsing_sub3 = np.array(hair_img.getdata(), dtype=np.float32)
        ##parsing_sub = np.hstack((parsing_sub1, parsing_sub2, parsing_sub1))
        ##parsing_sub = parsing_sub.reshape(1, 3, CROP_HEIGHT, CROP_WIDTH)-mean
        img_net_in = np.hstack((img_in, mean_in)) #1,4,h,w
        ##hair_net_in = img_in #1,3,h,w
        ##parsing_net_in = np.hstack((img_in, parsing_sub))*scale #1,6,h,w

        img_net_out = self.caffe_net.forward(**{self.caffe_net.inputs[0]: img_net_in})
        ##hair_net_out = self.caffe_net_hair.forward(**{self.caffe_net_hair.inputs[0]: hair_net_in})
        ##parsing_net_out = self.caffe_net_parsing.forward(**{self.caffe_net_parsing.inputs[0]: parsing_net_in})

        img_out = img_net_out[self.caffe_net.outputs[-1]]  #1, 1, 238, 188
        ##hair_out = hair_net_out[self.caffe_net_hair.outputs[-1]] #1, 1, 238, 188
        ##parsing_total_out = parsing_net_out[self.caffe_net_parsing.outputs[-1]] #1, 3, 125, 100
        ##parsing_out = parsing_total_out[0][1] * 255

        #debug visualize
        #net_img = self.arrayToGreyImg(img_out[0][0])
        #hair_img = self.arrayToGreyImg(hair_out[0][0])

        #net_img.save("net_img.png")
        #hair_img.save("hair_img.png")
        ##
        ##parsing_img = cv.fromarray(parsing_out)
        ##parsing_resize_img = cv.fromarray(np.zeros((2*PARSING_HEIGHT, 2*PARSING_WIDTH), dtype=np.float32))
        ##cv.Resize(parsing_img, parsing_resize_img) #enlarge

        ##parsing_possible_mat = np.array(parsing_resize_img[6:6+OUTPUT_HEIGHT, 6:6+OUTPUT_HEIGHT])
        #parsing_possible_mat = cv2.GaussianBlur(parsing_possible_mat, (3,3), 500, 500)
        ##parsing_possible_mat = cv2.GaussianBlur(parsing_possible_mat, (3,3), 500)
        ##cv2.erode(parsing_possible_mat, parsing_possible_mat, np.zeros((7,7), np.int8) )

        res = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype = np.int)
        for h in range(OUTPUT_HEIGHT):
            for w in range(OUTPUT_WIDTH):
                ##p = parsing_possible_mat[h][w] / 255
                ##res[h][w] = np.int( p*hair_out[0][0][h][w] + (1-p)*img_out[0][0][h][w] )
                res[h][w] = np.int(img_out[0][0][h][w])
                if (res[h][w] > 255):
                    res[h][w] = 255
                if (res[h][w] < 0):
                    res[h][w] = 0

        res_img = Image.fromarray(np.uint8(res))
        return res_img

    
    def arrayToGreyImg(self, arr):
        (height, width) = np.shape(arr)

        for h in range(height):
            for w in range(width): 
                if (arr[h][w] > 255):
                    arr[h][w] = 255
                if (arr[h][w] < 0):
                    arr[h][w] = 0

        img = Image.fromarray(np.uint8(arr))
        return img

    def crop2(self, filename):
        img = cv2.imread(filename)
        img = img[42:42+188, 27:27+143]
        cv2.imwrite('xixi.png',img)
        #cv2.imshow('xixi.png',img)
        #res.show()
        #return res
        #return img
        return 1


