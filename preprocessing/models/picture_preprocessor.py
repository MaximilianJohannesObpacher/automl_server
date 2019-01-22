from django.db import models

from preprocessing.models.audio_preprocessor import AudioPreprocessor


class PicturePreprocessor(AudioPreprocessor):
	class Meta:
		proxy = True