import pickle
from autokeras.utils import pickle_from_file


import sklearn
from django.db import transaction

from automl_systems.shared import load_ml_data
from training_server.models.validation_result import ValidationResult


def predict(conf):
	try:
		if conf.model.framework == 'auto_sklearn' or  conf.model.framework == 'tpot':
			with open(conf.model.model_path, 'rb') as f:
				my_model = pickle.load(f)
			x, y = load_ml_data(conf.model.validation_data_filename, conf.model.validation_labels_filename, True, conf.model.make_one_hot_encoding_task_binary)
		elif conf.model.framework == 'auto_keras':
			my_model = pickle_from_file(conf.model.model_path)
		else:
			print('notimpl (epic fail)')

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