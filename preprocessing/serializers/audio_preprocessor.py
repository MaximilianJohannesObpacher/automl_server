from rest_framework import serializers

from preprocessing.models.audio_preprocessor import AudioPreprocessor


class AudioPreprocessorSerializer(serializers.ModelSerializer):
	class Meta:
		fields = '__all__'
		model = AudioPreprocessor