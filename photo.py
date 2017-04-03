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
import detect_eye2

sys.path.append("/Users/zhaotsuchikaqin/caffe/python")
sys.path.append("/usr/local/Cellar/opencv/2.4.12/lib/python2.7/site-packages")
import caffe


PI = 3.1415926
EYE_DISTANCE = 50
EYE_EDGE_DISTANCE_X = 53
EYE_EDGE_DISTANCE_Y = 100

CROP_WIDTH = 155
CROP_HEIGHT = 200

OUTPUT_HEIGHT = 188
OUTPUT_WIDTH = 143


class Photo():

    caffe_net = caffe.Net("model/s2p/deploy2_6layers.prototxt","model/s2p/model_iter_400000.caffemodel",1)


    def origin(self,filename):
        img = Image.open(filename)

        return img
    def process(self, img):
        scale = 0.003921569
        mean = 128
        mean_file = "model/s2p/photo_mean.png"
        mean_img = Image.open(mean_file)
        
        mean_in = np.array(mean_img.getdata(), dtype=np.float32)
        mean_in = mean_in.reshape(1, 1, CROP_HEIGHT, CROP_WIDTH)-mean
        img_in = np.array(img.getdata(), dtype=np.float32)
        print img_in.shape
        
        img_in = np.transpose(img_in)
        print img_in.shape
        
        img_in = img_in.reshape(1, 1, CROP_HEIGHT, CROP_WIDTH)- mean
        
        
        img_net_in = np.hstack((img_in, mean_in)) #1,2,h,w
        

        img_net_out = self.caffe_net.forward(**{self.caffe_net.inputs[0]: img_net_in})
        
        img_out = img_net_out[self.caffe_net.outputs[-1]]  #1, 1, 238, 188
       
        
        
        net_img = self.arrayToGreyImg(img_out[0][0])
       
        
        #net_img.save("net_img.png")
       
        res = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype = np.int)
        for h in range(OUTPUT_HEIGHT):
            for w in range(OUTPUT_WIDTH):
                #p = parsing_possible_mat[h][w] / 255
                #res[h][w] = np.int( p*hair_out[0][0][h][w] + (1-p)*img_out[0][0][h][w] )
                res[h][w] = np.int(img_out[0][0][h][w])
                if (res[h][w] > 255):
                    res[h][w] = 255
                if (res[h][w] < 0):
                    res[h][w] = 0           
        res_img = Image.fromarray(np.uint8(res))
        #res_img.save("hehe.png")
        return res_img    
    #transform a arr to a grey pil image
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




    def point_rot(point, affineMat):

    #    print point.shape, affineMat.shape, affineMat[0,2], point[1]
        res = np.array([-1,-1])
        res[0] = affineMat[0,0]*point[1] + affineMat[0,1]*point[0] + affineMat[0,2]
            
        res[1] = affineMat[1,0]*point[1] + affineMat[1,1]*point[0] + affineMat[1,2]

        return res

    # crop to resized face image
    def crop(self, filename):
        img = Image.open(filename)
        print "haha"
        #print('Facepp Recognize result:', eyeDetector.faceppDetect(filename))
        #print('Opencv Recognize result:', eyeDetector.opencvDetect(filename))
        #result = eyeDetector.opencvDetect(filename)
        result = detect_eye2.caffeDetect(filename)
        print "xixi"
        res = Image.fromarray(result)
        print "haha"
        #res.show()
        print "hehe"
        return res
        #return img

        '''
    def point_rot(self, point, center, angle):
        rx = point['x']-center[0]
        ry = center[1]-point['y']
        rdis = math.sqrt(rx*rx+ry*ry)

        r_angle = math.atan(ry/rx)*180/PI

        if rx<0:
            r_angle = r_angle + 180
  

        n_angle = r_angle + angle

        res = {}
        res['x'] = center[0]+rdis*math.cos(n_angle*PI/180)
        res['y'] = center[1]-rdis*math.sin(n_angle*PI/180)

        return res

    # crop to resized face image
    def crop(self, filename):
        img = Image.open(filename)
        
        #print('Facepp Recognize result:', eyeDetector.faceppDetect(filename))
        #print('Opencv Recognize result:', eyeDetector.opencvDetect(filename))
        
        result = detect_eye2.caffeDetect(filename)
        lefteye = result['lefteye']
        righteye = result['righteye']
        

        print result
        
        dis_x = righteye['x'] - lefteye['x']
        dis_y = lefteye['y'] - righteye['y']

        if dis_x == dis_y:
            raise Exception("not human image")

        if dis_x == 0:
            angle = 90 if (lefteye['y'] > righteye['y']) else -90
        else:
            angle = math.atan(dis_y/dis_x)*180/PI

        center = (img.size[0]/2, img.size[1]/2)
        
        img = img.rotate(-angle)
        lefteye = self.point_rot(lefteye, center ,-angle)
        righteye = self.point_rot(righteye, center ,-angle)
        
        dis_x = righteye['x'] - lefteye['x']
        dis_y = lefteye['y'] - righteye['y']
        distance = math.sqrt(dis_x*dis_x+dis_y*dis_y)

        
        print "zero"
        scale = EYE_DISTANCE/distance

        img = img.resize( 
            (int(scale*img.size[0]), int(scale*img.size[1])) )

        lefteye['x'] = lefteye['x']*scale
        lefteye['y'] = lefteye['y']*scale
        righteye['x'] = righteye['x']*scale
        righteye['y'] = righteye['y']*scale
        print "2"
        
        min_x = int(lefteye['x']-EYE_EDGE_DISTANCE_X)
        min_y = int(lefteye['y']-EYE_EDGE_DISTANCE_Y)
        min_x = 0 if (min_x<0) else min_x
        min_y = 0 if (min_y<0) else min_y

        #max_x = int(righteye['x']+EYE_EDGE_DISTANCE_X)
        #max_y = int(righteye['y']+EYE_EDGE_DISTANCE_Y)

        #max_x = img.size[0] - 1 if (max_x < img.size[0]) else max_x
        #max_y = img.size[1] - 1 if (max_y < img.size[1]) else max_y

        max_x = min_x+CROP_WIDTH if ( min_x+CROP_WIDTH<img.size[0] ) else img.size[0]
        max_y = min_y+CROP_HEIGHT if ( min_y+CROP_HEIGHT<img.size[1] ) else img.size[1]
        img = img.crop((min_x, min_y, max_x, max_y))
        offset_x = int( (CROP_WIDTH-max_x+min_x) / 2)
        offset_y = int( (CROP_HEIGHT-max_y+min_y) / 2)
        img = img.resize((CROP_WIDTH, CROP_HEIGHT))
        img.show()
        

        
        #res = np.ones((CROP_HEIGHT, CROP_WIDTH, 1), dtype=np.uint8)*255
        #res[offset_y:offset_y+max_y-min_y, offset_x:offset_x+max_x-min_x, 0] = img
        #res[offset_y:offset_y+max_y-min_y, offset_x:offset_x+max_x-min_x, 1] = img 
        #res[offset_y:offset_y+max_y-min_y, offset_x:offset_x+max_x-min_x, 2] = img  
        #res = Image.fromarray(res)
        return img
    # transform photo to sketch image
'''
   
