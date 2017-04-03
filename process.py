import sys
import os
import types
import cv2
imgList = os.listdir('./sketch')
for x in imgList:
	print x
	if x[0]=='.':
		continue
	img = cv2.imread('./sketch/'+x)
	img = img[26:26+188,27:27+143]
	cv2.imwrite('./sketchbase/'+x,img)
	
	


'''
	if x[0]=='A':
		if len(x)==12:
			if x[3]=='0' and x[4]=='0':
				img=cv2.imread('./sketchdatabase/'+x)

				#cv2.imshow('hehe.png',img)
				cv2.imwrite('./sketchdatabase/'+'AR_'+x[5]+'.png',img)
				#cv2.waitKey()
				os.remove('./sketchdatabase/'+x)
'''

