import glob
import os
from shutil import copyfile

import numpy
import pandas as pd

from automl_server.settings import AUTO_ML_DATA_PATH, AUTO_ML_MODELS_PATH


def load_ml_data(data_filename, labels_filename, one_hot_encoded, transform_to_binary):
	x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, data_filename.replace(':', '_')))
	y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, labels_filename.replace(':', '_')))
	print('loaded')

	if one_hot_encoded:
		print('reformat!')
		return reformat_data(x), make_one_hot_encoding_binary(y) if transform_to_binary else make_one_hot_encoding_categorical(y)
	else:
		print('x,y')
		return x, y
		# TODO how to handel for keras: return x, make_one_hot_encoding_binary(y) if transform_to_binary else make_one_hot_encoding_categorical(y)

def save_ml_data(data_filename, labels_filename):
	x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, data_filename))
	y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, labels_filename))
	x = reformat_data(x)
	y = make_one_hot_encoding_binary(y)
	numpy.save(AUTO_ML_DATA_PATH+'/1'+data_filename, x)
	numpy.save(AUTO_ML_DATA_PATH+'/1'+labels_filename, y)
	return x, y

def reformat_data(x):
	print('Reformat started')
	nsamples = len(x)
	d2_npy = x.reshape((nsamples, -1))
	print(str(len(d2_npy)))
	return d2_npy


def make_one_hot_encoding_categorical(y):
	# replacing one_hot_encoding with letters for each category.
	labels = []
	labels_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] # make this dynamical depending on len onf labels
	for label in y:
		i = 0
		while i < 10:
			if label[i] == 1:
				labels.append(labels_replace[i])
				break
			i += 1
	print('Reformat finnished')
	return labels

def make_one_hot_encoding_binary(y):
	labels = []
	# Assuming state correct comes in first
	for label in y:
		i = 0
		if label[0] == 1:
			labels.append(1)
		else:
			labels.append(0)
	print('Reformat finnished')
	return labels


def get_file_path(features_name, labels_name):
	folders = []
	folder = ''

	for name in [features_name, labels_name]:
		if name.split('.'):
			filetype = name.split('.').last()
			if filetype == 'pkl':
				folder = '/pickle/'
			elif filetype == 'npy':
				folder = '/numpy/'
			elif filetype == 'csv':
				folder = '/csv/'
			else:
				print('Filetype not supported')  # TODO Throw error for filetype not supported

			folders.append(folder)
	return AUTO_ML_DATA_PATH + folders[0] + features_name, AUTO_ML_DATA_PATH + folders[1] + features_name


def load_from_folder(features_path, labels_path):
	files = []

	for path in [features_path, labels_path]:
		if os.path.exists(path):
			if path[-3:]=='pkl':
				files.append(pd.read_pickle(path))
			elif path[-3:]=='npy':
				files.append(numpy.load(path))
			elif path[-3:]=='csv':
				files.append(pd.read_csv(path))

	return files[0], files[1]


def file_loader(features_name, labels_name):

	features_path, labels_path = get_file_path(features_name, labels_name)

	return load_from_folder(features_path, labels_path)

def load_resnet50_model():
	from keras.applications import resnet50

	model = resnet50.ResNet50(include_top=True, weights='imagenet')
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	return model

