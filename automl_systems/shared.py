import os

import numpy

from automl_server.settings import AUTO_ML_DATA_PATH


def load_training_data(data_filename, labels_filename, reformat_required):
	x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, data_filename))  # size might crash it.
	y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, labels_filename))
	print('loaded')

	if reformat_required:
		return reformat_data(x, y)
	else:
		return x, make_one_hot_encoding_categorical(y)


def reformat_data(x, y):
	print('Reformat started')
	nsamples = len(x)
	d2_npy = x.reshape((nsamples, -1))
	print(str(len(d2_npy)))
	y = make_one_hot_encoding_categorical(y)
	print(str(len(y)))
	return d2_npy, y

def make_one_hot_encoding_categorical(y):
	# replacing one_hot_encoding with letters for each category.
	labels = []
	labels_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
	for label in y:
		i = 0
		while i < 10:
			if label[i] == 1:
				labels.append(labels_replace[i])
				break
			i += 1
	print('Reformat finnished')
	return labels
