import datetime
import glob
import os
import random

import numpy
from scipy.io.wavfile import read

from automl_server.settings import AUTO_ML_DATA_PATH
from automl_systems.shared import make_categorical_binary


def audio_to_image(path, height=192, width=192):
	print('shizzle')
	#signal, sr = lr.load(path, res_type='kaiser_fast')
	#hl = signal.shape[0] // (width * 1.1)
	#spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
	#img = lr.logamplitude(spec) ** 2
	#start = (img.shape[1] - width) // 2
	#return img[:, start:start + width]

def transform_all_audio_files_to_npy(transform_config):
	features_array = []
	labels_array = []

	try:

		for filepath in glob.iglob(AUTO_ML_DATA_PATH + transform_config.folder_name + '**/*.wav', recursive=True):
			features, label = audio_to_npy(filepath)
			features_array.append(features)
			labels_array.append(label)

		features_labels = list(zip(features_array, labels_array))

		random.shuffle(features_labels)
		features_array, labels_array = zip(*features_labels)

		print('Before:' + str(numpy.unique(labels_array, return_counts=True)))
		split_point = int(len(features_array) * 0.25)
		validation_features = numpy.array(features_array[:split_point])
		training_features = numpy.array(features_array[split_point:])
		validation_labels = labels_array[:split_point]
		training_labels = labels_array[split_point:]

		print('After: ' + str(numpy.unique(validation_labels, return_counts=True)) + ' other: ' +  str(numpy.unique(training_labels, return_counts=True)))

		print(str(len(features_array)) +'   '+ str(len(validation_features)) +'   '+ str(len(training_features)) +'   '+ str(len(labels_array)))

		timestamp = str(datetime.datetime.now())
		numpy.save(AUTO_ML_DATA_PATH + '/npy/training_features_' + str(timestamp) + '.npy', numpy.array(training_features))
		numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_' + str(timestamp) + '.npy', numpy.array(training_labels))
		numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_features_' + str(timestamp) + '.npy', numpy.array(validation_features))
		numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_labels_' + str(timestamp) + '.npy', numpy.array(validation_labels))

		if transform_config.transform_categorical_to_binary:
			training_labels_binary = make_categorical_binary(training_labels, transform_config.binary_true_name)
			validation_labels_binary = make_categorical_binary(validation_labels, transform_config.binary_true_name)

			numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(timestamp) + '.npy', training_labels_binary)
			(AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(timestamp) + '.npy', validation_labels_binary)

		transform_config.status = 'success'
		transform_config.save()

	except Exception as e:
		transform_config.additional_remarks = e
		transform_config.status = 'fail'
		transform_config.save()


def audio_to_npy(filepath):
	a = read(os.path.join(filepath))
	features = numpy.array(a[1], dtype=int)
	label = filepath.split('/')[-2]
	return features, label
