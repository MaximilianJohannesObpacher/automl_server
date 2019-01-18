from celery import Celery
from django.contrib.postgres.fields import ArrayField
from django.db import models


class FileReformater(models.Model):
	#AUTO_KERAS = 'auto_keras'
	#AUTO_SKLEARN = 'auto_sklearn'
	#TPOT = 'tpot'
#
	#ALGORITHM_CHOICES = (
	#	(AUTO_KERAS, 'Auto-keras'),
	#	(AUTO_SKLEARN, 'Auto-sklearn'),
	#	(TPOT, 'TPOT'),
	#)

	SUCCESS = 'success'
	FAIL = 'fail'

	STATUS_CHOICES = (
		(SUCCESS, 'Success'),
		(FAIL, 'Fail')
	)

	APACHE_PARQUET = 'parquet'
	PNG_IMAGE = 'png'
	WAV_AUDIO = 'wav'

	INPUT_CHOICES = (
		(APACHE_PARQUET, 'parquet'),
		(PNG_IMAGE, 'png'),
		(WAV_AUDIO, 'wav')
	)

	NUMPY_ARRAY = 'npy'
	CSV = 'csv'

	OUTPUT_CHOICES = (
		(NUMPY_ARRAY, 'npy'),
		(CSV, 'csv'),
	)

	# MULTICLASS_CLASSIFICATION = 'mlc'
	# BINARY_CLASSIFICATION = 'bic'
	# REGRESSION = 'REG'
	# TIME_SERIES = 'TS'

	#TASK_CHOICES = (
	#	(MULTICLASS_CLASSIFICATION, 'MULTICLASS_CLASSIFICATION'),
	#	(BINARY_CLASSIFICATION, 'BINARY_CLASSIFICATION'),
	#	(REGRESSION, 'REGRESSION'),
	#	(TIME_SERIES, 'TIME_SERIES')
	#)


	status = models.CharField(choices=STATUS_CHOICES, max_length=32, help_text='status of the training', null=True, blank=True)
	input_file_format = models.CharField(choices=INPUT_CHOICES, max_length=16, help_text='format of the input data')
	output_file_format = models.CharField(choices=OUTPUT_CHOICES, max_length=16, help_text='format of the output data')
	input_feature_file_name = models.CharField(max_length=256, help_text='name of the input file originating from the file type root folder')
	input_labels_file_name = models.CharField(max_length=256, help_text='name of the input labels originating from the file type root folder')

	input_validation_feature_file_name = models.CharField(max_length=256,
	                                           help_text='name of the input validation file originating from the file type root folder')
	input_validation_labels_file_name = models.CharField(max_length=256,
	                                          help_text='name of the input validation labels originating from the file type root folder')

	# task_choices = models.CharField(choices=TASK_CHOICES, max_length=32, help_text='what do you want to use the data for?')
