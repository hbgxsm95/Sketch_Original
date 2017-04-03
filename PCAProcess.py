import numpy as np
from PIL import Image
import time
import csv
import sys
import os
NUM = 6
NEWROW = 66
NEWCOL = 50
FPIMG_s = 'imagedatabase/'
#FPIMG_s = 'sketchdatabase/'
FPIMG_s2 = 'sketchbase/'
FPIMG_r = 'imageresdatabase/'
PCA_path = 'PCARunner/'
FNIMGs_csv = 'ImagePath_s.csv'
FNIMGr_csv = 'ImagePath_r.csv'
FNSM_csv = 'SampleMatrix.csv'
EIGENVALUE_csv = 'EigenValue.csv'
PCAFile_csv = 'PCAMatrix.csv'
MeanFile_csv = 'MeanMatrix.csv'
EigenFile_csv = 'EigenMatrix.csv'
PNG = 'png'
BMP = 'bmp'
TEST = 'test/'

def scale(arr):
	narr = np.empty((NEWROW, NEWCOL), dtype = np.double)
	kROW = arr.shape[0] * 1.0 / NEWROW
	kCOL = arr.shape[1] * 1.0 / NEWCOL
	for x in range(NEWROW):
		for y in range(NEWCOL):
			narr[x][y] = arr[x * kROW][y * kCOL]
	return narr

#write the filepath of images to the csv file,
#fp_img is the filepath of images, 
#fp is filepath of the PCA document,
#fn_csv is the filename of the csv file.

def writeImgPath(fps_img, fps_img2,fpr_img, fp, fns_csv, fnr_csv):
	csvFiles = open(fp + fns_csv, "wb")
	csvFiler = open(fp + fnr_csv, "wb")
	writers = csv.writer(csvFiles, quoting = csv.QUOTE_MINIMAL)
	writerr = csv.writer(csvFiler, quoting = csv.QUOTE_MINIMAL)
	imgList = os.listdir('./' + fps_img)
	listSize = len(imgList)
	if(listSize > 0):
		for x in imgList:
			writers.writerow([fps_img + x])
		for y in imgList:
			writerr.writerow([fpr_img + y[:-3] + BMP])
	else:
		print 'ERROR: No image in the filepath'
	imgList2 = os.listdir('./' + fps_img2)
	listSize2 = len(imgList2)
	if(listSize2 > 0):
		for x in imgList2:
			writers.writerow([fps_img2 + x])
	else:
		print 'ERROR: No image in the filepath'
	csvFiles.close()
	csvFiler.close()
	return listSize+listSize2

	

def writeNewImgPath(fpr_img):
	
	csvFiler = open(PCA_path + FNIMGr_csv, "ab")
	
	writerr = csv.writer(csvFiler, quoting = csv.QUOTE_MINIMAL)
	
	writerr.writerow([FPIMG_r+fpr_img])
	
	csvFiler.close()

#read the images to create a sampleMatrix,
#fp is filepath of the PCA document, 
#fn_csv is the filename of the csv file which record the images filepath
#num is the number of images

def readImgtoMatrix(fp, fn_csv, num):
	matrix = np.empty((num, NEWROW * NEWCOL), dtype = np.double)
	csvFile = file(fp + fn_csv, "rb")
	reader = csv.reader(csvFile)
	for line in reader:
		arr = scale(np.asarray(Image.open(line[0]).convert('L'))).reshape(1, NEWROW * NEWCOL)
		matrix[reader.line_num - 1] = arr
	print 'TOTAL: %dimages'%reader.line_num
	return matrix

#write the matrix to a csv file
#fp is filepath of the PCA document, 
#fn_csv is the filename of the csv file which you want to write down
#matrix is the matrix which you want to write down

def writeMatrix(fp, fn_csv, matrix):
	csvFile = open(fp + fn_csv, "wb")
	writer = csv.writer(csvFile, quoting = csv.QUOTE_MINIMAL)
	for x in range(matrix.shape[0]):
		writer.writerow(matrix[x])
	csvFile.close()

def writeNewMatrix(fp, fn_csv, matrix):
	csvFile = open(fp + fn_csv, "ab")
	writer = csv.writer(csvFile, quoting = csv.QUOTE_MINIMAL)
	writer.writerow(matrix)
	csvFile.close()

#read the matrix from csv file to create a matrix
#fp is filepath of the PCA document, 
#fn_csv is the filename of the csv file which you want to read

def readMatrix(fp, fn_csv):
	csvFile = file(fp + fn_csv, "rb")
	csvFile1 = file(fp + fn_csv, "rb")
	reader = csv.reader(csvFile)
	num = 0
	for line in reader:
		num += 1
	csvFile.close()
	matrix = np.empty((num, len(line)), dtype = np.double)
	reader = csv.reader(csvFile1)
	for line in reader:
		matrix[reader.line_num - 1] = np.asarray(line)
	csvFile1.close()
	return matrix

def PCA(sample):
	mean = sample.mean(axis = 0)
	src = sample.copy()
	src -= mean
	print 'COV'
	COV = src.T.dot(src) / src.shape[0]
	print 'EIG'
	v, m = np.linalg.eig(COV)
	index = np.argsort(v)[::-1]
	print 'SORT'
	m = m[:,index]
	v = v[index]
	esum = 0
	src = src[0:471,:]
	print 'F to K'
	for x in range(v.shape[0]):
		esum += v[x]
		if(esum / v.sum() >= 0.95):
			NEWm = m[:,:x + 1].astype(np.double)
			break
	print 'GET'
	PCAMatrix = (NEWm.T.dot(src.T)).T
	return PCAMatrix, mean, NEWm

def process():
	st = time.time()
	sampleNum = writeImgPath(FPIMG_s,FPIMG_s2,FPIMG_r, PCA_path, FNIMGs_csv,FNIMGr_csv)
	print 'STEP: 1'
	SampleMatrix = readImgtoMatrix(PCA_path, FNIMGs_csv, sampleNum)
	print 'STEP: 2'
	writeMatrix(PCA_path, FNSM_csv, SampleMatrix)
	print 'STEP: 3'
	SampleMatrix = readMatrix(PCA_path, FNSM_csv)
	print 'STEP: 4'
	m, meanMatrix, EigenMatrix = PCA(SampleMatrix)
	print 'STEP: 5'
	writeMatrix(PCA_path, PCAFile_csv, m)
	writeMatrix(PCA_path, MeanFile_csv, meanMatrix.reshape(1, NEWROW * NEWCOL))
	writeMatrix(PCA_path, EigenFile_csv, EigenMatrix)
	PCAm = readMatrix(PCA_path, PCAFile_csv)
	print 'FIN'
	print time.time() - st

def showImage(fp, fn_csv, num, index):
	csvFile = fp + fn_csv
	imgList = []
	#readcsv =[0,1,2,3,4,5]
	#for line in reader:
	#	if(reader.line_num == num[count] + 1):
	#		imgList.append(line[0])
	#		count += 1
	#	if(count >= NUM):
	#		break
	#print imgList.__len__()		
	reList = []
	#print imgList
	#for x in index:
	#	reList.append(imgList[-x])
	#return reList
	for dex in range(0,6):
		count = 0
		with open(csvFile,'rb') as f:
			for i in f.readlines():
				if(count == num[dex]):
					field = i.split()[0]
					reList.append(field)
					break
				count = count+1;
	#return reList
	return reList;

def match(img):
	arr = scale(np.asarray(img.convert('L'))).reshape(1, NEWROW * NEWCOL)
	#Image.fromarray(scale(np.asarray(img.convert('L')))).show()
	mean = readMatrix(PCA_path, MeanFile_csv)
	pca = readMatrix(PCA_path, PCAFile_csv)
	eigen = readMatrix(PCA_path, EigenFile_csv)
	sample = readMatrix(PCA_path, FNSM_csv)
	arr -= mean
	arrpca = (eigen.T.dot(arr.T)).T
	arrpcaMatrix = np.empty(pca.shape, dtype = np.double)
	for x in range(pca.shape[0]):
		arrpcaMatrix[x] = arrpca
	num = np.argsort(((arrpcaMatrix - pca) ** 2).sum(axis = 1))[:NUM]
	#print num
	#index = np.argsort(num)
	index = np.asarray([2,3,4,1,5,0])
	#print index
	#num = num[index]
	#print num
	return showImage(PCA_path, FNIMGr_csv, num, index)

def uploadnew(img):
	arr = scale(np.asarray(img.convert('L'))).reshape(1, NEWROW * NEWCOL)
	#Image.fromarray(scale(np.asarray(img.convert('L')))).show()
	mean = readMatrix(PCA_path, MeanFile_csv)
	pca = readMatrix(PCA_path, PCAFile_csv)
	eigen = readMatrix(PCA_path, EigenFile_csv)
	sample = readMatrix(PCA_path, FNSM_csv)
	arr -= mean
	arrpca = (eigen.T.dot(arr.T)).T
	arrpcaMatrix = np.empty(pca.shape, dtype = np.double)
	arrpcaMatrix[0]=arrpca
	writeNewMatrix(PCA_path, PCAFile_csv, arrpcaMatrix[0])

if(__name__ == '__main__'):
	process()
	#i = Image.open('AR_030_s.png')
	#i.show()
	#imgList = match(i)
	#print imgList
	#for x in imgList:
	#	Image.open(x).show()
	#uploadnew(i)