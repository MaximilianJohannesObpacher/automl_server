from django.db import models

from preprocessing.models.file_preprocessor import FilePreprocessor


class PicturePreprocessor(FilePreprocessor):
	output_image_dimens = models.IntegerField(default=128)
