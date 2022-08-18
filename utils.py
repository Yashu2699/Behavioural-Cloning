import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def getName(filePath):
	return filePath.split('\\')[-1]

def importDataInfo(path):
	columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
	data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)

	# .apply() applies the getName function to all the images in data['Center']
	data['Center'] = data['Center'].apply(getName)
	return data

def balanceData(data, display=False):
	nBins = 31
	samplesPerBin = 1000
	hist, bins = np.histogram(data['Steering'], nBins)

	if display:
		center = (bins[:-1] + bins[1:]) * 0.5
		plt.bar(center, hist, width=0.06)

		# We truncate the steering values at 0.0 because there are many values
		plt.plot((-1,1), (samplesPerBin, samplesPerBin))
		plt.show()

	# Visit every bin and keep only samplesPerBin samples
	
	removeIndexList = []
	for j in range(nBins):
		binDataList = []
		for i in range(len(Data['Steering'])):
			if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
				binDataList[i]
		binDataList = shuffle(binDataList)
		binDataList = binDataList[samplesPerBin:]
		removeIndexList.extend(binDataList)

	print('Removed Images:', len(removeIndexList))
	data.drop(data.index[removeIndexList], inplace=True)
	print('Remaining Images:', len(data))
	
	if display:
		hist, _ = np.histogram(data['Steering'], nBins)
		plt.bar(center, hist, width=0.06)
		plt.show()

	return data


# function to convert images, steering to an array
def loadData(path, data):
	imagesPath = []
	steering = []

	for i in range(len(data)):
		indexedData = data.iloc[i]
		imagesPath.append(os.path.join(path, 'IMG', indexedData[0]))
		steering.append(float(indexedData[3]))
	imagesPath = np.asarray(imagesPath)
	steering = np.asarray(steering)

	return imagesPath, steering


# Image Augmentation
def augmentImage(imgpath, steering):
	# We use mpimg because we deal with rgb images and cv2 deals with bgr
	img = mpimg.imread(imgpath)

	# Translation
	# Shift is random 
	if np.random.rand() < 0.5:
		pan = iaa.Affine(translate_percent={
						'x' : (-0.1, 0.1),
						'y' : (-0.1, 0.1)
			})
		img = pan.augment_image(img)

	# Zoom 
	if np.random.rand() < 0.5:
		zoom = iaa.Affine(scale=(1,1.2))
		img = zoom.augment_image(img)

	# Brightness (0 - dark, 1 - Normal, >1 - bright)
	if np.random.rand() < 0.5:
		brightness = iaa.Multiply((0.4,1.2))
		img = brightness.augment_image(img)

	# Flip
	if np.random.rand() < 0.5:
		img = cv2.flip(img, 1)
		steering = -steering

	return img, steering

# Preprocessing Images
def preprocess(img):

	# Retaining only the road segment of the image
	img = img[60:135, :, :]

	# Changing the color space form rgb to yuv
	# lane lines are much more visible in yuv space
	img = cv2.cvtColot(img, cv2.COLOR_RGB2YUV)

	# Blur the img
	img = cv2.GaussianBlur(img, (3,3), 0)

	# Resize the image
	img = cv2.resize(img, (200,66))

	# Normalizing the image values
	img = img / 255

	return img


# Batch Generator
def batchGen(imagesPath, steeringList, batchsize, trainFlag=True):

	while True:

		imgBatch = []
		steeringBatch = []

		for i in range(batchsize):
			index = random.randint(0, len(imagesPath)-1)

			if trainFlag:
				img, steering = augmentImage(imagesPath[index], steeringList[index])
			else:
				img = mpimg.imread(imagesPath[index])
				steering = steeringList[index]
			img = preprocess(img)
			imgBatch.append(img)
			steeringBatch.append(steering)

		yield (np.asarray(imgBatch), np.asarray(steeringBatch))


def create_model():

	model = Sequential()
	model.add(Convolution2D(24, (5,5), (2,2), input_shape=(66,200,3), activation='elu'))
	model.add(Convolution2D(36, (5,5), (2,2), activation='elu'))
	model.add(Convolution2D(48, (5,5), (2,2), activation='elu'))
	model.add(Convolution2D(64, (5,5), activation='elu'))
	model.add(Convolution2D(64, (5,5), activation='elu'))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))

	model.compile(Adam(lr=0.0001), loss='mse')

