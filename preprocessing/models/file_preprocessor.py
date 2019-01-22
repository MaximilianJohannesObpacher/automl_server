from django.db import models

from preprocessing.models.audio_preprocessor import FilePreprocessor


class AudioPreprocessor(FilePreprocessor):
	folder_name = models.CharField(max_length=256, default='/wav/', blank=True, null=True)
	transform_categorical_to_binary = models.BooleanField(default=False)
	binary_true_name = models.CharField(max_length=256, null=True, blank=True, default='no_fat_behavior')
