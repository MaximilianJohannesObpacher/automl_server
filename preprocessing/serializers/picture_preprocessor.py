from rest_framework import serializers

from preprocessing.models.picture_preprocessor import PicturePreprocessor


class PicturePreprocessorSerializer(serializers.ModelSerializer):
	class Meta:
		fields = '__all__'
		model = PicturePreprocessor
