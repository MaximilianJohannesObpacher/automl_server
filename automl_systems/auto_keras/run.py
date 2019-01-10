import datetime
import os
import pickle
import time

import autokeras as ak
import numpy
from autokeras import ImageClassifier

from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from automl_systems.shared import load_training_data
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
		x_train = numpy.load(os.path.join(AUTO_ML_DATA_PATH, 'merged_folds_training_x0.1.npy'))
		y_train = numpy.load(os.path.join(AUTO_ML_DATA_PATH, 'merged_folds_training_y0.1.npy'))

		# x_test = numpy.load(os.path.join(AUTO_ML_DATA_PATH, 'merged_folds_validation_x.npy'))
		# y_test = numpy.load(os.path.join(AUTO_ML_DATA_PATH, 'merged_folds_validation_y.npy'))

		## replacing one_hot_encoding with letters for each category.
		labels = []
		labels_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

		for label in y_train:
			i = 0
			while i < 10:
				if label[i] == 1:
					labels.append(labels_replace[i])
					break
				i += 1
		y_train = labels
		labels = []

		#for label in y_test:
		#	i = 0
		#	while i < 10:
		#		if label[i] == 1:
		#			labels.append(labels_replace[i])
		#			break
		#		i += 1
		#y_test = labels

		clf = ImageClassifier(verbose=True)
		clf.fit(x_train, y_train, time_limit=5 * 60)
		#clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
		#y = clf.evaluate(x_test, y_test)
		#print(y)
		print("Fitting Success!!!")


		#dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_keras' + str(datetime.datetime.now()) + '.dump')
#
		#x, y = load_training_data(auto_keras_config.input_data_filename, auto_keras_config.labels_filename, False)
#
		#print(len(x))
		#print(len(y))
		#print('before training init')
		#model = ImageClassifier(verbose=auto_keras_config.verbose)
		#print('before training start')
		#start = time.time()
		#model.fit(x, y, time_limit=auto_keras_config.time_limit)
		#end = time.time()
#
		##x = model.show_models()
		#results = {"ensemble": x}
#
		#print('pickling')
		## storing the best performer
		#with open(dump_file, 'wb') as f:
		#	model.export_autokeras_model(f)
#
		#auto_keras_config.training_time = round(end-start, 2)
		#auto_keras_config.status = 'success'
		#auto_keras_config.model_path = dump_file
		#auto_keras_config.save()
		#print('Status final ' +auto_keras_config.status)

	except Exception as e:
		end = time.time()
		#if 'start' in locals():
		#	print('failed after:' + str(end-start))
		#	auto_keras_config.training_time = round(end-start, 2)

		auto_keras_config.status = 'fail'
		auto_keras_config.additional_remarks = e
		auto_keras_config.save()