import os

import numpy
from django.db import models
from scipy.io.wavfile import read

from preprocessing.models.file_preprocessor import FilePreprocessor


class AudioPreprocessor(FilePreprocessor):
	FLOAT = 'float'
	INT = 'int'

	FILE_FORMAT_CHOICES = (
		(FLOAT, 'float'),
		(INT, 'int')
	)

	output_data_format = models.CharField(max_length=256, blank=True, null=True, choices=FILE_FORMAT_CHOICES)

	def save_audio_as_npy(self, filepath):
		a = read(os.path.join(filepath))

		if self.output_data_format == 'float':
			dtype = float
		else:
			dtype = int

		features = numpy.array(a[1], dtype=dtype)
		label = filepath.split('/')[-2]
		return features, label
