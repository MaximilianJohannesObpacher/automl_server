from celery import Celery
from django.contrib.postgres.fields import ArrayField
from django.db import models


# from automl_server.automl_systems.autosklearn.run import train, add
# from automl_server.automl_systems.tpot.run import train as train_tpot
from preprocessing.models.audio_preprocessor import FilePreprocessor


class AlgorithmConfig(models.Model):
	AUTO_KERAS = 'auto_keras'
	AUTO_SKLEARN = 'auto_sklearn'
	TPOT = 'tpot'

	ALGORITHM_CHOICES = (
		(AUTO_KERAS, 'Auto-keras'),
		(AUTO_SKLEARN, 'Auto-sklearn'),
		(TPOT, 'TPOT'),
	)

	IN_PROGRESS = 'in_progress'
	SUCCESS = 'success'
	FAIL = 'fail'
	WAITING = 'waiting'

	STATUS_CHOICES = (
		(WAITING, 'Waiting for thread'),
		(IN_PROGRESS, 'In progress'),
		(SUCCESS, 'Success'),
		(FAIL, 'Fail')
	)

	PREPROCESSING_OBJECT = 'preprocessing_job'
	FILENAME = 'filename'

	LOADING_CHOICES = (
		(PREPROCESSING_OBJECT, 'Preprocessing Job Output'), # selection if binary or classification
		(FILENAME, 'filename') # 4 filename fields for preventing validating on training set accidentially.
	)

	BINARY_CLASSIFICATION = 'binary_classification'
	MULTICLASS_CLASSIFICATION = 'multiclass_classification'

	TASK_CHOICES = (
		(BINARY_CLASSIFICATION, 'binary classification'),
		(MULTICLASS_CLASSIFICATION, 'multiclass classification')
	)

	training_name = models.CharField(max_length=256, default='Unnamed')
	framework = models.CharField(max_length=24, choices=ALGORITHM_CHOICES)
	model_path = models.CharField(null=True, blank=True, help_text='Path to the model', max_length=256)
	status = models.CharField(null=True, blank=True, help_text='Status of the training', choices=STATUS_CHOICES,
	                          max_length=32)
	date_trained = models.DateTimeField(auto_now=True)
	training_triggered = models.BooleanField(default=False,
	                                         help_text='Helper Flag for defining which config should be updateable (which one has not yet been trained)')
	additional_remarks = models.CharField(null=True, blank=True, max_length=2048,
	                                      help_text='Additional Information about the training. E.g. Information about failed trainings are logged here in case a training fails!')
	training_data_filename = models.CharField(default='merged_folds_training_x.npy', max_length=256, null=True, blank=True,
	                                          help_text='Filename or path to the training data file originating from ml_data folder')
	training_labels_filename = models.CharField(default='merged_folds_training_y.npy', max_length=256, null=True, blank=True,
	                                            help_text='Filename or path to the training labels file originating from ml_data folder')
	validation_data_filename = models.CharField(default='merged_folds_validation_x.npy', max_length=256, null=True, blank=True,
	                                            help_text='Filename or path to the validation data file originating from ml_data folder')
	validation_labels_filename = models.CharField(default='merged_folds_validation_y.npy', max_length=256, null=True, blank=True,
	                                              help_text='Filename or path to the validation labels file originating from ml_data folder')
	training_time = models.CharField(blank=True, null=True, max_length=128,
	                                 help_text='training time until completion or interrupt (in seconds)')
	input_one_hot_encoded = models.BooleanField(default=False, help_text='Handle one hot encoding by labeling the data with string instead to be accepted.')
	make_one_hot_encoding_task_binary = models.BooleanField(default=False, help_text='Only possible for categorical data with one-hot-encoding. If the flag is checked, the first option is assumed to be option 0 and all options afterwards are assuemd to be option 1')
	freeze_results = models.BooleanField(default=False, help_text='Click this to avoid tempering with the results by making the training immutable after executing it.')
	load_files_from = models.CharField(choices=LOADING_CHOICES, help_text='Decide wether you want to load your files from a filepath or a grab the output files of a preprocessing job you triggered before', max_length=32, default='filename')
	task_type = models.CharField(choices=TASK_CHOICES, blank=True, null=True, help_text='If undefined we automatically perform multiclass classification', max_length=32) # only relevant if load input from files
	preprocessing_object = models.ForeignKey(FilePreprocessor, null=True, blank=True)

	def __str__(self):
		return str(self.model_path)
