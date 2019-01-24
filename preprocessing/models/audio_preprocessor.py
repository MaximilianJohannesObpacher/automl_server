from django.db import models

from preprocessing.models.file_preprocessor import FilePreprocessor


class AudioPreprocessor(FilePreprocessor):
	file_format = models.CharField(max_length=256, blank=True, null=True)
