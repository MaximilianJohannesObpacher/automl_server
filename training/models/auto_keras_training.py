import datetime
import os
import time

import numpy
from autokeras import ImageClassifier
from django.db import models

from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from training.models import AutoMlTraining


class AutoKerasTraining(AutoMlTraining):
	time_limit = models.IntegerField(null=True, blank=True)
	verbose = models.BooleanField(default=True)

	def train(self):

		self.status = 'in_progress'
		self.save()
		# Storing save location for models

		try:
			dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_keras' + str(datetime.datetime.now()) + '.h5')

			x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, self.training_data_filename))
			y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, self.training_labels_filename))
			# x, y = load_ml_data(auto_keras_training.training_data_filename, auto_keras_config.training_labels_filename, False, auto_keras_config.make_one_hot_encoding_task_binary)

			# TODO this might not work on low ram machines work, but array has to be 3d
			if self.preprocessing_object.input_data_type == 'wav':
				array4d = []
				i = 0
				for datapoint in x:
					x_3d = datapoint.reshape((172, 128, 10)).transpose()
					array4d.append(x_3d)
					i += 1
				x = numpy.array(array4d)

			clf = ImageClassifier(verbose=self.verbose)

			start = time.time()
			clf.fit(x, y, time_limit=self.time_limit)
			end = time.time()

			# clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
			# y = clf.evaluate(x_test, y_test)

			print("Fitting Success!!!")

			# storing the best performer
			clf.export_autokeras_model(dump_file)
			print('saved!')
			#
			self.training_time = round(end - start, 2)
			self.status = 'success'
			self.model_path = dump_file
			self.save()
			print('Status final ' + self.status)

		except Exception as e:
			end = time.time()
			if 'start' in locals():
				print('failed after:' + str(end - start))
				self.training_time = round(end - start, 2)

			self.status = 'fail'
			self.additional_remarks = e
			self.save()
