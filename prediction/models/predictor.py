import pickle

import cv2
import os
from os import read

import librosa
import numpy
from PIL import Image
from django.db import models
from rest_framework.exceptions import ValidationError

from shared import reformat_data
from training.models import AutoMlTraining


class Predictor(models.Model):
	training = models.ForeignKey(AutoMlTraining, null=True, blank=True)
	file = models.FileField(null=True, blank=True, upload_to='ml_data/prediction_files/')
	result = models.CharField(null=True, blank=True, max_length=256)

	def predict(self):

		filepath = self.file.file.file.name

		if self.file.name.split('.')[1:][0] == 'wav':
			a = read(os.path.join(filepath))
			features = numpy.array(a[1], dtype=int)

		elif self.file.name.split('.')[1:][0] == 'png':
			img = Image.open(filepath)
			img.resize((128, 128))
			img.save(filepath)

			features = numpy.array([cv2.imread(filepath)])

			# first order difference, computed over 9-step window
			features[0, :, :, 1] = librosa.feature.delta(features[0, :, :, 0])

			# for using 3 dimensional array to use ResNet and other frameworks
			features[0, :, :, 2] = librosa.feature.delta(features[0, :, :, 1])

			features = numpy.transpose(features, (0, 2, 1, 3))
			features = reformat_data(features)
		else:
			return ValidationError('File not in supported format. Supported formats are png and wav.')

		with open(self.training.model_path, 'rb') as f:
			my_model = pickle.load(f)

		y_pred = my_model.predict(features)

		self.result = y_pred[0]
		self.save()