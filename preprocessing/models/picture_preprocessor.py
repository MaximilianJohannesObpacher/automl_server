from django.db import models

from preprocessing.models.audio_preprocessor import AudioPreprocessor
from preprocessing.models.file_preprocessor import FilePreprocessor


class PicturePreprocessor(FilePreprocessor):
	bands = models.CharField(max_length=256, blank=True, null=True)
	frames = models.CharField(max_length=256, blank=True, null=True)
