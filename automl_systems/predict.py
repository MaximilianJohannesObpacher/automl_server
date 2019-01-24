import os
import pickle

import numpy
import sklearn

from autokeras.utils import pickle_from_file

from automl_server.settings import AUTO_ML_DATA_PATH
from automl_systems.shared import load_ml_data, reformat_data


def predict(conf):
	try:
		if conf.model.framework == 'auto_sklearn' or  conf.model.framework == 'tpot':
			with open(conf.model.model_path, 'rb') as f:
				my_model = pickle.load(f)

			x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, conf.model.training_data_filename))
			y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, conf.model.training_labels_filename))

			if conf.model.preprocessing_object.input_data_type == 'png':
				x = reformat_data(x)

		elif conf.model.framework == 'auto_keras':
			my_model = pickle_from_file(conf.model.model_path)
			x, y = load_ml_data(conf.model.validation_data_filename, conf.model.validation_labels_filename, False, conf.model.make_one_hot_encoding_task_binary)
		else:
			print('notimpl (epic fail)')

		print('about to pred.')
		y_pred = my_model.predict(x)
		print('about to acc')

		if conf.scoring_strategy == 'accuracy':
			score=sklearn.metrics.accuracy_score(y, y_pred)
		elif(conf.scoring_strategy == 'precision'):
			score=sklearn.metrics.average_precision_score(y, y_pred)
		elif(conf.scoring_strategy == 'roc_auc'):
			score=sklearn.metrics.roc_auc_score(y, y_pred)
		else:
			score = 0
			print('epic fail! no Strat applied')

		print('savy!')
		conf.status = 'success'
		conf.score = str(round(score,4))
		conf.save()

	except Exception as e:
		conf.status = 'fail'
		conf.additional_remarks = e
		conf.save()
