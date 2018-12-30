import autosklearn.classification
import sklearn.datasets
import sklearn.metrics

def train(config):
	try:
		X, y = sklearn.datasets.load_digits(return_X_y=True)
		X_train, X_test, y_train, y_test = \
		        sklearn.model_selection.train_test_split(X, y, random_state=1)
		automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=60)
		automl.fit(X_train, y_train)
		y_hat = automl.predict(X_test)
		print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

		config.status = 'SUCCESS'
		config.save()
		config.additional_remarks = "Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat)
		config.save()

	except Exception as e:
		config.status = 'fail'
		config.additional_remarks = e
		config.save()