import datetime
import os
import pickle
import time

import autokeras as ak
import numpy
from autokeras import ImageClassifier

from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from automl_systems.shared import load_ml_data
from training.models.auto_keras_training import AutoKerasTraining

def train(auto_keras_training_id):
	auto_keras_training = AutoKerasTraining.objects.get(id=auto_keras_training_id)

	auto_keras_training.status = 'in_progress'
	auto_keras_training.save()
	# Storing save location for models

	try:
		dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_keras' + str(datetime.datetime.now()) + '.h5')

		x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, auto_keras_training.training_data_filename))
		y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, auto_keras_training.training_labels_filename))
		# x, y = load_ml_data(auto_keras_training.training_data_filename, auto_keras_config.training_labels_filename, False, auto_keras_config.make_one_hot_encoding_task_binary)

		# TODO this might not work on low ram machines work, but array has to be 3d
		if auto_keras_training.preprocessing_object.input_data_type == 'wav':
			array4d = []
			i=0
			for datapoint in x:
				x_3d = datapoint.reshape((172,128,10)).transpose()
				array4d.append(x_3d)
				i+=1
			x = numpy.array(array4d)


		clf = ImageClassifier(verbose=auto_keras_training.verbose)

		start = time.time()
		clf.fit(x, y, time_limit=auto_keras_training.time_limit)
		end = time.time()

		#clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
		#y = clf.evaluate(x_test, y_test)

		print("Fitting Success!!!")

		# storing the best performer
		clf.export_autokeras_model(dump_file)
		print('saved!')
#
		auto_keras_training.training_time = round(end-start, 2)
		auto_keras_training.status = 'success'
		auto_keras_training.model_path = dump_file
		auto_keras_training.save()
		print('Status final ' + auto_keras_training.status)

	except Exception as e:
		end = time.time()
		if 'start' in locals():
			print('failed after:' + str(end-start))
			auto_keras_training.training_time = round(end-start, 2)

		auto_keras_training.status = 'fail'
		auto_keras_training.additional_remarks = e
		auto_keras_training.save()
