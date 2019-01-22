from django.db import models

from preprocessing.models.file_preprocessor import FilePreprocessor


class AudioPreprocessor(FilePreprocessor):

	transform_categorical_to_binary = models.BooleanField(default=False, help_text='should the data be labeled binary as well?')

	# as mixin field:
	binary_true_name = models.CharField(max_length=256, null=True, blank=True, default='perfect_condition', help_text='if binary transform categorical data to binary is true, all files in folder labeled with this name will be labeled as True while all other data will be labeled as false.')