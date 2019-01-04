import autosklearn.classification
import sklearn.datasets
import sklearn.metrics

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime

import autosklearn.classification
import numpy
import os
import pandas
import sys
import six.moves.cPickle as pickle
from celery import shared_task
from sklearn.model_selection import train_test_split

from training_server.celery import app
from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from training_server.models import AutoSklearnConfig

target_name = 'jordan'


def ingest():
	training_data = pandas.read_csv(os.path.join(AUTO_ML_DATA_PATH,
	                                             'numerai_training_data.csv'), header=0)
	tournament_data = pandas.read_csv(os.path.join(AUTO_ML_DATA_PATH,
	                                               'numerai_tournament_data.csv'), header=0)
	features = [f for f in list(training_data) if 'feature' in f]
	x = training_data[features]
	y = training_data['target_' + target_name]
	x_tournament = tournament_data[features]
	ids = tournament_data['id']
	return (x, y, x_tournament, ids)

	# be careful to persist the data and how they were split for training, so we do not validate on training data

@app.task()
def train(auto_sklearn_config_id):
	auto_sklearn_config = AutoSklearnConfig.objects.get(id=auto_sklearn_config_id)
	print('Autosklearnconfig object: ' + auto_sklearn_config.status)

	auto_sklearn_config.status = 'in_progress'
	auto_sklearn_config.save()
	# Storing save location for models
	print('Autosklearnconfig object2: ' + auto_sklearn_config.status)
	try:
		dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_sklearn' + str(datetime.datetime.now()) + '.dump')

		# TODO load klaidis proposed numpy arrays
		x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, 'merged_folds_training_x.npy')) # size might crash it.
		y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, 'merged_folds_training_y.npy'))


		nsamples = len(x)
		d2_npy = x.reshape((nsamples, -1))

		# replacing one_hot_encoding with letters for each category.
		labels = []
		labels_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
		for label in y:
			i = 0
			while i < 10:
				if label[i] == 1:
					labels.append(labels_replace[i])
					break
				i += 1

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
			logging_config=auto_sklearn_config.logging_config
		)
		model.fit(d2_npy, labels)
		print(model.show_models())

		x = model.show_models()
		results = {"ensemble": x}

		print('pickling')
		# storing the best performer
		with open(dump_file, 'wb') as f:
			pickle.dump(results, f)

		print('writing config status')
		auto_sklearn_config.status = 'success'
		auto_sklearn_config.model_path = dump_file
		auto_sklearn_config.save()
		print('Status final ' +auto_sklearn_config.status)

	except Exception as e:
		auto_sklearn_config.status = 'fail'
		auto_sklearn_config.additional_remarks = e
		auto_sklearn_config.save()


def predict(model, x_tournament, ids):
	eps = sys.float_info.epsilon
	y_prediction = model.predict_proba(x_tournament)
	results = numpy.clip(y_prediction[:, 1], 0.0 + eps, 1.0 - eps)
	results_df = pandas.DataFrame(data={'probability_' + target_name: results})
	joined = pandas.DataFrame(ids).join(results_df)
	joined.to_csv(os.path.join(AUTO_ML_DATA_PATH,
	                           'prediction_' + target_name + '.csv'), index=False, float_format='%.16f')


def main():
	x, y, x_tournament, ids = ingest()
	model = train(x, y)
	predict(model, x_tournament.copy(), ids)


if __name__ == '__main__':
	main()
