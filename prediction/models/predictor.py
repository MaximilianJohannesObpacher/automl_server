import pickle

import cv2
import os
from os import read

import librosa
import numpy
from PIL import Image
from django.db import models
from django.db.models import Max
from rest_framework.exceptions import ValidationError

from preprocessing.models.file_preprocessor import FilePreprocessor
from shared import reformat_data
from training.models import AutoMlTraining


class Predictor(models.Model):

	BINARY_CLASSIFICATION = 'binary_classification'
	MULTICLASS_CLASSIFICATION = 'multiclass_classification'

	TASK_CHOICES = (
		(BINARY_CLASSIFICATION, 'binary classification'),
		(MULTICLASS_CLASSIFICATION, 'multiclass classification')
	)

	training = models.ForeignKey(AutoMlTraining, null=True, blank=True)
	file = models.FileField(null=True, blank=True, upload_to='ml_data/prediction_files/')
	result = models.CharField(null=True, blank=True, max_length=256)
	machine_id = models.CharField(max_length=256, null=True, blank=True)
	task_type = models.CharField(choices=TASK_CHOICES, blank=True, null=True, help_text='If undefined we automatically perform multiclass classification', max_length=32) # only relevant if load input from files

	def predict(self):
		filepath = self.file.file.file.name
		input_data_type = self.file.name.split('.')[1:][0]

		if self.training:
			with open(self.training.model_path, 'rb') as f:
				my_model = pickle.load(f)
		else:
			# if preprocessing job exists, figure out which datatype the preprocessing job object had
			try:
				# getting all preprocessors which are linked to a model from this machine with this type
				machine_preprecessors = FilePreprocessor.objects.filter(input_data_type=input_data_type, machine_id=self.machine_id)

				# Getting
				training = AutoMlTraining.objects.filter(preprocessing_object__in=machine_preprecessors, status='success', task_type=self.task_type, validator__scoring_strategy='accuracy').order_by('-validator__score').first()
				with open(training.model_path, 'rb') as f:
					my_model = pickle.load(f)

				self.training = training
			except:
				raise ValidationError("Either training or machine id has to be provided!")

		if input_data_type == 'wav':
			file = read(os.path.join(filepath))
			features = numpy.array(file[1], dtype=int)

		elif input_data_type == 'png':
			file = Image.open(filepath)
			file.resize((128, 128))
			file.save(filepath)

			features = numpy.array([cv2.imread(filepath)])

			# first order difference, computed over 9-step window
			features[0, :, :, 1] = librosa.feature.delta(features[0, :, :, 0])

			# for using 3 dimensional array to use ResNet and other frameworks
			features[0, :, :, 2] = librosa.feature.delta(features[0, :, :, 1])

			features = numpy.transpose(features, (0, 2, 1, 3))
			features = reformat_data(features)
		else:
			return ValidationError('File not in supported format. Supported formats are png and wav.')

		y_pred = my_model.predict(features)

		self.result = y_pred[0]
