import datetime
import glob
import os
import random

import cv2
import librosa
import numpy
from scipy.io.wavfile import read

from automl_server.settings import AUTO_ML_DATA_PATH
from preprocessing.file_preprocessing.categorical_to_binary import make_categorical_binary


def transform_all_audio_files_to_npy(transform_config, is_audio):
	features_array = []
	labels_array = []

	try:

		# get all files and put them in a features and a labels array
		if is_audio:
			for filepath in glob.iglob(AUTO_ML_DATA_PATH + transform_config.input_folder_name + '**/*.wav', recursive=True):
				features, label = audio_to_npy(filepath)
				features_array.append(features)
				labels_array.append(label)
		else:
			for filepath in glob.iglob(AUTO_ML_DATA_PATH + transform_config.input_folder_name + '**/*.png', recursive=True):
				features, label = label_picture(filepath)
				features_array.append(features)
				labels_array.append(label)

		print('features_reformat success')
		print(str(features_array))
		features_labels = list(zip(features_array, labels_array))

		# shuffling
		random.shuffle(features_labels)
		features_array, labels_array = zip(*features_labels)

		print('Before:' + str(numpy.unique(labels_array, return_counts=True)))

		# splitting in training and validation;
		features_array = numpy.array(features_array)

		if not is_audio:
			print('not audio')
			print(len(features_array))
			features_array = features_array / 255.0
			print('after features array devide')

			for i in range(len(features_array)):
				print(i)
				# first order difference, computed over 9-step window
				features_array[i, :, :, 1] = librosa.feature.delta(features_array[i, :, :, 0])

				# for using 3 dimensional array to use ResNet and other frameworks
				features_array[i, :, :, 2] = librosa.feature.delta(features_array[i, :, :, 1])

			features_array = numpy.transpose(features_array, (0, 2, 1, 3))

		split_point = int(len(features_array) * 0.25)
		validation_features = features_array[:split_point]
		training_features = features_array[split_point:]
		validation_labels = labels_array[:split_point]
		training_labels = labels_array[split_point:]

		print('After: ' + str(numpy.unique(validation_labels, return_counts=True)) + ' other: ' +  str(numpy.unique(training_labels, return_counts=True)))

		# saving as npy arrays
		timestamp = str(datetime.datetime.now())

		print('trying to save audio')
		numpy.save(AUTO_ML_DATA_PATH + '/npy/training_features_' + str(timestamp) + '.npy',
			           numpy.array(training_features))
		numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_features_' + str(timestamp) + '.npy',
			           numpy.array(validation_features))

		transform_config.training_features_path = AUTO_ML_DATA_PATH + '/npy/training_features_' + str(timestamp) + '.npy'
		transform_config.evaluation_features_path = AUTO_ML_DATA_PATH + '/npy/validation_features_' + str(timestamp) + '.npy'

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

			numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(timestamp) + '.npy', training_labels_binary)
			numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(timestamp) + '.npy', validation_labels_binary)
			transform_config.training_labels_path_binary = AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(timestamp) + '.npy'
			transform_config.evaluation_labels_path_binary = AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(timestamp) + '.npy'

		transform_config.status = 'success'
		print(transform_config.training_features_path)
		return transform_config

	except Exception as e:
		transform_config.additional_remarks = e
		transform_config.status = 'fail'
		transform_config.save()


def audio_to_npy(filepath):
	a = read(os.path.join(filepath))
	features = numpy.array(a[1], dtype=float)
	label = filepath.split('/')[-2]
	return features, label

def label_picture(filepath):
	img = cv2.imread(filepath)
	label = filepath.split('/')[-2]
	return img, label

def transform_my_audio_files_to_npy():
	features_array = []
	labels_array = []
	is_audio = False
	input_folder_name ='/png/'
	transform_categorical_to_binary = True
	binary_true_name = 'perfect_condition'

	try:

		# get all files and put them in a features and a labels array
		if is_audio:
			for filepath in glob.iglob(AUTO_ML_DATA_PATH + input_folder_name + '**/*.wav', recursive=True):
				features, label = audio_to_npy(filepath)
				features_array.append(features)
				labels_array.append(label)
		else:
			for filepath in glob.iglob(AUTO_ML_DATA_PATH + input_folder_name + '**/*.png', recursive=True):
				features, label = label_picture(filepath)
				features_array.append(features)
				labels_array.append(label)

		print('features_reformat success')
		print(str(features_array))
		features_labels = list(zip(features_array, labels_array))

		# shuffling
		random.shuffle(features_labels)
		features_array, labels_array = zip(*features_labels)

		print('Before:' + str(numpy.unique(labels_array, return_counts=True)))

		# splitting in training and validation;
		features_array = numpy.array(features_array)

		if not is_audio:
			print('not audio')
			print(len(features_array))
			features_array = features_array / 255.0
			print('after features array devide')

			for i in range(len(features_array)):
				print(i)
				# first order difference, computed over 9-step window
				features_array[i, :, :, 1] = librosa.feature.delta(features_array[i, :, :, 0]) # TODO this array is only 3 dimens, we need 4 dimens.

				# for using 3 dimensional array to use ResNet and other frameworks
				features_array[i, :, :, 2] = librosa.feature.delta(features_array[i, :, :, 1])

			features_array = numpy.transpose(features_array, (0, 2, 1, 3))

		split_point = int(len(features_array) * 0.25)
		validation_features = features_array[:split_point]
		training_features = features_array[split_point:]
		validation_labels = labels_array[:split_point]
		training_labels = labels_array[split_point:]

		print('After: ' + str(numpy.unique(validation_labels, return_counts=True)) + ' other: ' +  str(numpy.unique(training_labels, return_counts=True)))

		# saving as npy arrays
		timestamp = str(datetime.datetime.now())

		print('trying to save audio')
		numpy.save(AUTO_ML_DATA_PATH + '/npy/training_features_' + str(timestamp) + '.npy',
			           numpy.array(training_features))
		numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_features_' + str(timestamp) + '.npy',
			           numpy.array(validation_features))

		training_features_path = AUTO_ML_DATA_PATH + '/npy/training_features_' + str(timestamp) + '.npy'
		evaluation_features_path = AUTO_ML_DATA_PATH + '/npy/validation_features_' + str(timestamp) + '.npy'

		numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_' + str(timestamp) + '.npy', training_labels)
		numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_labels_' + str(timestamp) + '.npy', validation_labels)

		training_labels_path = AUTO_ML_DATA_PATH + '/npy/training_labels_' + str(
			timestamp) + '.npy'
		evaluation_labels_path = AUTO_ML_DATA_PATH + '/npy/validation_labels_' + str(
			timestamp) + '.npy'

		# optional saving classification task as binary task as well.
		if transform_categorical_to_binary:
			training_labels_binary = make_categorical_binary(training_labels, binary_true_name)
			validation_labels_binary = make_categorical_binary(validation_labels, binary_true_name)

			numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(timestamp) + '.npy', training_labels_binary)
			numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(timestamp) + '.npy', validation_labels_binary)
			training_labels_path_binary = AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(timestamp) + '.npy'
			evaluation_labels_path_binary = AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(timestamp) + '.npy'

		print('success')

	except Exception as e:
		print(e)