from django.db import models

from training_server.models import FileReformater


class AudioReformater(FileReformater):
	folder_name = models.CharField(max_length=256, default='/wav/', blank=True, null=True)
	transform_categorical_to_binary = models.BooleanField(default=False)
	binary_true_name = models.CharField(max_length=256, null=True, blank=True, default='no_fat_behavior')
