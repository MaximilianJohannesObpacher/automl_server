import os

import numpy

from automl_server.settings import AUTO_ML_DATA_PATH


def load_ml_data(data_filename, labels_filename, reformat_required, transform_to_binary):
	x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, data_filename))  # size might crash it.
	y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, labels_filename))
	print('loaded')

	if reformat_required:
		return reformat_data(x), make_one_hot_encoding_binary(y) if transform_to_binary else make_one_hot_encoding_categorical(y)
	else:
		return x, make_one_hot_encoding_binary(y) if transform_to_binary else make_one_hot_encoding_categorical(y)


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
	labels_replace = ['a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
	for label in y:
		i = 0
		while i < 10:
			if label[i] == 1:
				labels.append(labels_replace[i])
				break
			i += 1
	print('Reformat finnished')
	return labels