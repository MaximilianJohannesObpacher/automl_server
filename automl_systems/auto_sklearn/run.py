import multiprocessing
import time

import autosklearn.classification
import billiard
import numpy
import sklearn.datasets
import sklearn.metrics

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime

import autosklearn.classification
import os
import six.moves.cPickle as pickle
from autosklearn.regression import AutoSklearnRegressor

from automl_systems.shared import load_ml_data, file_loader, reformat_data, load_resnet50_model
from training.celery import app
from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from training.models import AutoSklearnTraining

@app.task()
def train(auto_sklearn_training_id):
	auto_sklearn_training = AutoSklearnTraining.objects.get(id=auto_sklearn_training_id)

	auto_sklearn_training.status = 'in_progress'
	auto_sklearn_training.save()
	# Storing save location for models

	try:

		dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_sklearn' + str(datetime.datetime.now()) + '.dump')

		x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, auto_sklearn_training.training_data_filename))
		y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, auto_sklearn_training.training_labels_filename))

		if auto_sklearn_training.preprocessing_object.input_data_type == 'png':
			x = reformat_data(x)

		model = autosklearn.classification.AutoSklearnClassifier(
			time_left_for_this_task=auto_sklearn_training.run_time,
			per_run_time_limit=auto_sklearn_training.per_instance_runtime,
			initial_configurations_via_metalearning=auto_sklearn_training.initial_configurations_via_metalearning,
			ml_memory_limit=auto_sklearn_training.memory_limit,
			ensemble_size=auto_sklearn_training.ensemble_size,
			ensemble_nbest=auto_sklearn_training.ensemble_nbest,
			seed=auto_sklearn_training.seed,
			include_estimators=auto_sklearn_training.include_estimators,
			exclude_estimators=auto_sklearn_training.exclude_estimators,
			include_preprocessors=auto_sklearn_training.include_preprocessors,
			exclude_preprocessors=auto_sklearn_training.exclude_preprocessors,
			resampling_strategy=auto_sklearn_training.resampling_strategy,
			tmp_folder=auto_sklearn_training.tmp_folder,
			output_folder=auto_sklearn_training.output_folder,
			delete_tmp_folder_after_terminate=auto_sklearn_training.delete_tmp_folder_after_terminate,
			delete_output_folder_after_terminate=auto_sklearn_training.delete_output_folder_after_terminate,
			shared_mode=auto_sklearn_training.shared_mode,
			smac_scenario_args=auto_sklearn_training.smac_scenario_args,
			logging_config=auto_sklearn_training.logging_config,
			)
		print('before training start')
		start = time.time()
		model.fit(x, y)
		end = time.time()
		print(model.show_models())
		# storing the best performer
		with open(dump_file, 'wb') as f:
			pickle.dump(model, f)

		auto_sklearn_training.training_time = round(end-start, 2)
		auto_sklearn_training.status = 'success'
		auto_sklearn_training.model_path = dump_file
		auto_sklearn_training.save()
		print('Status final ' +auto_sklearn_training.status)

	except Exception as e:
		end = time.time()
		if 'start' in locals():
			print('failed after:' + str(end-start))
			auto_sklearn_training.training_time = round(end-start, 2)

		auto_sklearn_training.status = 'fail'
		auto_sklearn_training.additional_remarks = e
		auto_sklearn_training.save()


def train_regression():
	dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_sklearn_regressor' + str(datetime.datetime.now()) + '.dump')

	features, outcome_slave , _ = file_loader('c99temp_train.snappy.csv')

	features = features.values
	outcome_slave = outcome_slave['tempBoardSLAVE'].values

	model = AutoSklearnRegressor(
		time_left_for_this_task=3600,
		per_run_time_limit=600,
	)
	model.fit(features, outcome_slave)

	with open(dump_file, 'wb') as f:
		pickle.dump(model, f)

