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
from training_server.celery import app
from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from training_server.models import AutoSklearnConfig

@app.task()
def train(auto_sklearn_config_id):
	auto_sklearn_config = AutoSklearnConfig.objects.get(id=auto_sklearn_config_id)

	auto_sklearn_config.status = 'in_progress'
	auto_sklearn_config.save()
	# Storing save location for models

	try:

		dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_sklearn' + str(datetime.datetime.now()) + '.dump')

		x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, auto_sklearn_config.training_data_filename))
		y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, auto_sklearn_config.training_labels_filename))

		if auto_sklearn_config.preprocessing_object.input_data_type == 'png':
			x = reformat_data(x)

		model = autosklearn.classification.AutoSklearnClassifier(
			time_left_for_this_task=auto_sklearn_config.run_time,
			per_run_time_limit=auto_sklearn_config.per_instance_runtime,
			initial_configurations_via_metalearning=auto_sklearn_config.initial_configurations_via_metalearning,
			ml_memory_limit=auto_sklearn_config.memory_limit,
			ensemble_size=auto_sklearn_config.ensemble_size,
			ensemble_nbest=auto_sklearn_config.ensemble_nbest,
			seed=auto_sklearn_config.seed,
			include_estimators=auto_sklearn_config.include_estimators,
			exclude_estimators=auto_sklearn_config.exclude_estimators,
			include_preprocessors=auto_sklearn_config.include_preprocessors,
			exclude_preprocessors=auto_sklearn_config.exclude_preprocessors,
			resampling_strategy=auto_sklearn_config.resampling_strategy,
			tmp_folder=auto_sklearn_config.tmp_folder,
			output_folder=auto_sklearn_config.output_folder,
			delete_tmp_folder_after_terminate=auto_sklearn_config.delete_tmp_folder_after_terminate,
			delete_output_folder_after_terminate=auto_sklearn_config.delete_output_folder_after_terminate,
			shared_mode=auto_sklearn_config.shared_mode,
			smac_scenario_args=auto_sklearn_config.smac_scenario_args,
			logging_config=auto_sklearn_config.logging_config,
			)
		print('before training start')
		start = time.time()
		model.fit(x, y)
		end = time.time()
		print(model.show_models())
		# storing the best performer
		with open(dump_file, 'wb') as f:
			pickle.dump(model, f)

		auto_sklearn_config.training_time = round(end-start, 2)
		auto_sklearn_config.status = 'success'
		auto_sklearn_config.model_path = dump_file
		auto_sklearn_config.save()
		print('Status final ' +auto_sklearn_config.status)

	except Exception as e:
		end = time.time()
		if 'start' in locals():
			print('failed after:' + str(end-start))
			auto_sklearn_config.training_time = round(end-start, 2)

		auto_sklearn_config.status = 'fail'
		auto_sklearn_config.additional_remarks = e
		auto_sklearn_config.save()


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

