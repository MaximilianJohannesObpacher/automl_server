import datetime
import glob
import os
import random

import numpy
from django.db import models
from scipy.io.wavfile import read

from automl_server.settings import AUTO_ML_DATA_PATH
from preprocessing.file_preprocessing.resize_images import resize_images, generate_image_array


class FilePreprocessorManager(models.Manager):
	def get_queryset(self):
		return super().get_queryset().filter(status='success')


class FilePreprocessor(models.Model):
	objects = FilePreprocessorManager()

	SUCCESS = 'success'
	FAIL = 'fail'

	STATUS_CHOICES = (
		(SUCCESS, 'Success'),
		(FAIL, 'Fail')
	)

	PNG = 'png'
	WAV = 'wav'

	datatype_choices = (
		(PNG, 'png'),
		(WAV, 'wav')
	)

	status = models.CharField(choices=STATUS_CHOICES, max_length=32, help_text='status of the training', null=True,
	                          blank=True)
	additional_remarks = models.CharField(null=True, blank=True, max_length=2048,
                                      help_text='Additional Information about the training. E.g. Information about failed trainings are logged here in case a training fails!')
	transform_categorical_to_binary = models.BooleanField(default=False, help_text='should the data be labeled binary as well?')
	training_features_path = models.CharField(max_length=256, null=True, blank=True)
	training_labels_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_features_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_labels_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_labels_path_binary = models.CharField(max_length=256, null=True, blank=True)
	training_labels_path_binary = models.CharField(max_length=256, null=True, blank=True)
	binary_true_name = models.CharField(max_length=256, null=True, blank=True, default='perfect_condition', help_text='if binary transform categorical data to binary is true, all files in folder labeled with this name will be labeled as True while all other data will be labeled as false.')
	input_folder_name = models.CharField(max_length=256, default='', blank=True, null=True)
	input_data_type = models.CharField(blank=True, null=True, choices=datatype_choices, max_length=32)
	preprocessing_name = models.CharField(max_length=255, null=True, blank=True)

	def __str__(self):
		return str(self.input_folder_name) + '_' + str(self.training_features_path)

	def audio_to_npy(self, filepath):
		a = read(filepath)

		if self.output_data_format == 'float':
			dtype = float
		else:
			dtype = int

		features = numpy.array(a[1], dtype=dtype)
		label = filepath.split('/')[-2]
		return features, label

	def make_categorical_binary(labels, true_name):
		bin_labels = []

		for label in labels:
			if label == true_name:
				bin_labels.append(1)
			else:
				bin_labels.append(0)

		return bin_labels

	def transform_media_files_to_npy(self, is_audio):
		features_array = []
		labels_array = []

		try:

			# get all files and put them in a features and a labels array
			if is_audio:
				print(is_audio)
				for filepath in glob.iglob(AUTO_ML_DATA_PATH + self.input_folder_name + '**/*.wav',
				                           recursive=True):
					features, label = self.audio_to_npy(filepath)
					features_array.append(features)
					labels_array.append(label)
			# case image
			else:
				resize_images(self.output_image_dimens)
				features_array, labels_array = generate_image_array()

			print(len(features_array))
			features_labels = list(zip(features_array, labels_array))

			#  shuffling
			random.shuffle(features_labels)
			features_array, labels_array = zip(*features_labels)

			print('Before:' + str(numpy.unique(labels_array, return_counts=True)))

			split_point = int(len(features_array) * 0.3)
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

			self.training_features_path = AUTO_ML_DATA_PATH + '/npy/training_features_' + str(
				timestamp) + '.npy'
			self.evaluation_features_path = AUTO_ML_DATA_PATH + '/npy/validation_features_' + str(
				timestamp) + '.npy'

			numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_' + str(timestamp) + '.npy', training_labels)
			numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_labels_' + str(timestamp) + '.npy', validation_labels)

			self.training_labels_path = AUTO_ML_DATA_PATH + '/npy/training_labels_' + str(
				timestamp) + '.npy'
			self.evaluation_labels_path = AUTO_ML_DATA_PATH + '/npy/validation_labels_' + str(
				timestamp) + '.npy'

			# optional saving classification task as binary task as well.
			if self.transform_categorical_to_binary:
				training_labels_binary = self.make_categorical_binary(training_labels, self.binary_true_name)
				validation_labels_binary = self.make_categorical_binary(validation_labels, self.binary_true_name)

				numpy.save(AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(timestamp) + '.npy',
				           training_labels_binary)
				numpy.save(AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(timestamp) + '.npy',
				           validation_labels_binary)
				self.training_labels_path_binary = AUTO_ML_DATA_PATH + '/npy/training_labels_bin_' + str(
					timestamp) + '.npy'
				self.evaluation_labels_path_binary = AUTO_ML_DATA_PATH + '/npy/validation_labels_bin_' + str(
					timestamp) + '.npy'

			self.status = 'success'
			print(self.training_features_path)
			self.save()
			return self

		except Exception as e:
			print(e)
			self.additional_remarks = e
			self.status = 'fail'
			self.save()