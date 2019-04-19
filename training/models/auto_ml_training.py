import os

import numpy
from celery import Celery
from django.contrib.postgres.fields import ArrayField
from django.db import models

from automl_server.settings import AUTO_ML_DATA_PATH
from preprocessing.models.audio_preprocessor import FilePreprocessor


class AutoMlTraining(models.Model):
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
	additional_remarks = models.CharField(null=True, blank=True, max_length=100000,
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
	label_one_hot_encoding_binary = models.BooleanField(default=False, help_text='Only possible for categorical data with one-hot-encoding. If the flag is checked, the first option is assumed to be option 0 and all options afterwards are assuemd to be option 1')
	freeze_results = models.BooleanField(default=False, help_text='Click this to avoid tempering with the results by making the training immutable after executing it.')
	load_files_from = models.CharField(choices=LOADING_CHOICES, help_text='Decide wether you want to load your files from a filepath or a grab the output files of a preprocessing job you triggered before', max_length=32, default='filename')
	task_type = models.CharField(choices=TASK_CHOICES, blank=True, null=True, help_text='If undefined we automatically perform multiclass classification', max_length=32) # only relevant if load input from files
	preprocessing_object = models.ForeignKey(FilePreprocessor, null=True, blank=True)
	machine_id = models.CharField(max_length=256, null=True, blank=True)
	model = models.FileField(upload_to='ml_models/', null=True, blank=True)

	def __str__(self):
		return str(self.model_path)

	def save(self, *args, **kwargs):
		self.link_correct_labels()
		super(AutoMlTraining, self).save(*args, **kwargs)

	def link_correct_labels(self):
		if self.load_files_from == 'preprocessing_job' and self.preprocessing_object:
			self.training_data_filename = self.preprocessing_object.training_features_path
			self.validation_data_filename = self.preprocessing_object.evaluation_features_path
			if self.task_type == 'binary_classification':
				self.training_labels_filename = self.preprocessing_object.training_labels_path_binary
				self.validation_labels_filename = self.preprocessing_object.evaluation_labels_path_binary

			else:
				self.training_labels_filename = self.preprocessing_object.training_labels_path
				self.validation_labels_filename = self.preprocessing_object.evaluation_labels_path

	def config_algorithm(self, akt):
		# load labels
		y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, akt.training_labels_filename))

		# find out the amount of different classes
		count_classes = len(numpy.unique(y))
		print('Count classes: ' + str(count_classes))

		if akt.framework == 'tpot':
			akt.run_time = akt.max_time_mins
		if akt.framework == 'auto_keras':
			akt.run_time = akt.time_limit

		akt.run_time = int(akt.run_time * count_classes - 1)

		# get filesize
		filesize = os.path.getsize(os.path.join(AUTO_ML_DATA_PATH, akt.training_data_filename))

		# filesize divided by one gigabyte
		akt.run_time = int(akt.run_time * filesize / 1000000000)

		# get representation
		representation = akt.preprocessing_object.input_data_type

		if representation == 'wav':
			akt.run_time = akt.run_time * 2

		akt.per_instance_runtime = int(akt.run_time / 10)

		if akt.framework == 'tpot':
			print('in tpot')
			akt.max_time_mins = akt.run_time * 2
			akt.max_eval_time_mins = akt.per_instance_runtime * 2
			akt.population_size = 4
			akt.generations = 3

		if akt.framework == 'auto-keras':
			print('in akeras')
			akt.time_limit = akt.run_time * 10
		return akt


	def save_labels(self, akt):
		if akt.load_files_from == 'preprocessing_job':
			akt.training_data_filename = akt.preprocessing_object.training_features_path
			akt.validation_data_filename = akt.preprocessing_object.evaluation_features_path
			if akt.task_type == 'binary_classification':
				akt.training_labels_filename = akt.preprocessing_object.training_labels_path_binary
				akt.validation_labels_filename = akt.preprocessing_object.evaluation_labels_path_binary
			else:
				akt.training_labels_filename = akt.preprocessing_object.training_labels_path
				akt.validation_labels_filename = akt.preprocessing_object.evaluation_labels_path
		return akt