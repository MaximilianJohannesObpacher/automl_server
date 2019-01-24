import cv2
import datetime
import glob
import os
import random

import numpy
from PIL import Image
from scipy.io.wavfile import read

from automl_server.settings import AUTO_ML_DATA_PATH
from preprocessing.file_preprocessing.categorical_to_binary import make_categorical_binary


def pad_trunc_seq_rewrite(x, max_len):
	if x.shape[1] < max_len:
		pad_shape = (x.shape[0], max_len - x.shape[1])
		pad = numpy.ones(pad_shape) * numpy.log(1e-8)
		x_new = numpy.hstack((x, pad))

	# no pad necessary - truncate
	else:
		x_new = x[:, 0:max_len]
	return x_new


def transform_all_audio_files_to_npy(transform_config, is_audio):
	features_array = []
	labels_array = []

	try:

		# get all files and put them in a features and a labels array
		if is_audio:
			for filepath in glob.iglob(AUTO_ML_DATA_PATH + transform_config.input_folder_name + '**/*.wav',
			                           recursive=True):
				features, label = audio_to_npy(filepath)
				features_array.append(features)
				labels_array.append(label)
		else:
			for filepath in glob.iglob(AUTO_ML_DATA_PATH + transform_config.input_folder_name + '**/*.png',
			                           recursive=True):
				features , label = label_picture(filepath)
				print('FTTTT:', features)

				features_array.append(features)
				labels_array.append(label)



		print('features_reformat success')
		print(len(features_array))
		features_labels = list(zip(features_array, labels_array))
		print('features_reformat success2')

		#  shuffling
		random.shuffle(features_labels)
		print('features_reformat success3')
		features_array, labels_array = zip(*features_labels)
		print('features_reformat success4')

		print('Before:' + str(numpy.unique(labels_array, return_counts=True)))

		split_point = int(len(features_array) * 0.25)
		validation_features = features_array[:split_point]
		training_features = features_array[split_point:]
		validation_labels = labels_array[:split_point]
		training_labels = labels_array[split_point:]

		print('After: ' + str(numpy.unique(validation_labels, return_counts=True)) + ' other: ' + str(
			numpy.unique(training_labels, return_counts=True)))

		# saving as npy arrays
		timestamp = str(datetime.datetime.now())

		print('trying to save audio')
		numpy.save(AUTO_ML_DATA_PATH + '/npy/training_features_' + str(timestamp) + '.npy',
		           numpy.array(training_features))
		numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_features_' + str(timestamp) + '.npy',
		           numpy.array(validation_features))

		transform_config.training_features_path = AUTO_ML_DATA_PATH + '/npy/training_features_' + str(
			timestamp) + '.npy'
		transform_config.evaluation_features_path = AUTO_ML_DATA_PATH + '/npy/validation_features_' + str(
			timestamp) + '.npy'

		numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_' + str(timestamp) + '.npy', training_labels)
		numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_labels_' + str(timestamp) + '.npy', validation_labels)

		transform_config.training_labels_path = AUTO_ML_DATA_PATH + '/npy/training_labels_' + str(
			timestamp) + '.npy'
		transform_config.evaluation_labels_path = AUTO_ML_DATA_PATH + '/npy/validation_labels_' + str(
			timestamp) + '.npy'

		# optional saving classification task as binary task as well.
		if transform_config.transform_categorical_to_binary:
			training_labels_binary = make_categorical_binary(training_labels, transform_config.binary_true_name)
			validation_labels_binary = make_categorical_binary(validation_labels, transform_config.binary_true_name)

			numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(timestamp) + '.npy',
			           training_labels_binary)
			numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(timestamp) + '.npy',
			           validation_labels_binary)
			transform_config.training_labels_path_binary = AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(
				timestamp) + '.npy'
			transform_config.evaluation_labels_path_binary = AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(
				timestamp) + '.npy'

		transform_config.status = 'success'
		print(transform_config.training_features_path)
		return transform_config

	except Exception as e:
		print(e)
		transform_config.additional_remarks = e
		transform_config.status = 'fail'
		transform_config.save()

def audio_to_npy(filepath):
	a = read(os.path.join(filepath))
	features = numpy.array(a[1], dtype=float)
	label = filepath.split('/')[-2]
	return features, label


def label_picture(filepath):
	img = cv2.resize(cv2.imread(filepath), (128,128))
	label = filepath.split('/')[-2]
	return img, label