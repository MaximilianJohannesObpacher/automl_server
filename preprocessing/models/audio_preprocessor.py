from os import read

import numpy
from django.db import models

from preprocessing.models.file_preprocessor import FilePreprocessor


class AudioPreprocessor(FilePreprocessor):
	FLOAT = 'float'
	INT = 'int'

	FILE_FORMAT_CHOICES = (
		(FLOAT, 'float'),
		(INT, 'int')
	)

	output_data_format = models.CharField(max_length=256, blank=True, null=True, choices=FILE_FORMAT_CHOICES)