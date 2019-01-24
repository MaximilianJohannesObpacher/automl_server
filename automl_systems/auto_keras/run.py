import datetime
import os
import pickle
import time

import autokeras as ak
import numpy
from autokeras import ImageClassifier

from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from automl_systems.shared import load_ml_data
from training_server.celery import app
from training_server.models.auto_keras_config import AutoKerasConfig

@app.task()
def train(auto_keras_config_id):
	print('auto_keras_config object: ' + str(auto_keras_config_id))
	auto_keras_config = AutoKerasConfig.objects.get(id=auto_keras_config_id)

	auto_keras_config.status = 'in_progress'
	auto_keras_config.save()
	# Storing save location for models

	try:
		dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_keras' + str(datetime.datetime.now()) + '.h5')

		print('Files to load: ' + auto_keras_config.training_data_filename)

		x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, auto_keras_config.training_data_filename))
		y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, auto_keras_config.training_labels_filename))
		# x, y = load_ml_data(auto_keras_config.training_data_filename, auto_keras_config.training_labels_filename, False, auto_keras_config.make_one_hot_encoding_task_binary)

		if auto_keras_config.preprocessing_object.input_data_type == 'wav':
			array4d = []
			i=0
			for datapoint in x:
				print(i)
				x_3d = datapoint.reshape((172,128,10)).transpose()
				array4d.append(x_3d)
				i+=1
			x = numpy.array(array4d)


		clf = ImageClassifier(verbose=auto_keras_config.verbose)

		start = time.time()
		clf.fit(x, y, time_limit=auto_keras_config.time_limit)
		end = time.time()

		#clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
		#y = clf.evaluate(x_test, y_test)

		print("Fitting Success!!!")

		# storing the best performer
		clf.export_autokeras_model(dump_file)
		print('saved!')
#
		auto_keras_config.training_time = round(end-start, 2)
		auto_keras_config.status = 'success'
		auto_keras_config.model_path = dump_file
		auto_keras_config.save()
		print('Status final ' + auto_keras_config.status)

	except Exception as e:
		end = time.time()
		if 'start' in locals():
			print('failed after:' + str(end-start))
			auto_keras_config.training_time = round(end-start, 2)

		auto_keras_config.status = 'fail'
		auto_keras_config.additional_remarks = e
		auto_keras_config.save()
