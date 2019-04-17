from rest_framework import serializers

from preprocessing.models.file_preprocessor import FilePreprocessor
from preprocessing.models.picture_preprocessor import PicturePreprocessor


class PicturePreprocessorSerializer(serializers.ModelSerializer):
	class Meta:
		fields = '__all__'
		model = PicturePreprocessor

	def create(self, validated_data):
		obj = PicturePreprocessor.objects.create(**validated_data)
		obj.input_folder_name = '/wav/'
		FilePreprocessor.transform_media_files_to_npy(obj, True)  # TODO Find out how to make async
		obj.input_data_type = 'wav'
		obj.save()
		return obj
