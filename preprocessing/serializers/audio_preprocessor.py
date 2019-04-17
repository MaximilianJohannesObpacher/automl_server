from rest_framework import serializers

from preprocessing.models.audio_preprocessor import AudioPreprocessor
from preprocessing.models.file_preprocessor import FilePreprocessor


class AudioPreprocessorSerializer(serializers.ModelSerializer):
	class Meta:
		fields = '__all__'
		model = AudioPreprocessor

	def create(self, validated_data):
		obj = AudioPreprocessor.objects.create(**validated_data)
		obj.input_folder_name = '/wav/'
		FilePreprocessor.transform_media_files_to_npy(obj, True)  # TODO Find out how to make async
		obj.input_data_type = 'wav'
		obj.save()
		return obj
