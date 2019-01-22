from celery import Celery
from django.contrib.postgres.fields import ArrayField
from django.db import models


class FilePreprocessor(models.Model):

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
	PICKLE = 'pkl'

	OUTPUT_CHOICES = (
		(NUMPY_ARRAY, 'npy'),
		(CSV, 'csv'),
		(PICKLE, 'pkl')
	)

	status = models.CharField(choices=STATUS_CHOICES, max_length=32, help_text='status of the training', null=True,
	                          blank=True)
	input_file_format = models.CharField(choices=INPUT_CHOICES, max_length=16, help_text='format of the input data')
	output_file_format = models.CharField(choices=OUTPUT_CHOICES, max_length=16, help_text='format of the output data')
	additional_remarks = models.CharField(null=True, blank=True, max_length=2048,
                                      help_text='Additional Information about the training. E.g. Information about failed trainings are logged here in case a training fails!')
