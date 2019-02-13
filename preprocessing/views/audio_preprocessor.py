from rest_framework import viewsets

from preprocessing.models.audio_preprocessor import AudioPreprocessor
from preprocessing.serializers.audio_preprocessor import AudioPreprocessorSerializer


class AudioPreprocessorViewSet(viewsets.ModelViewSet):
	queryset = AudioPreprocessor.objects.all()
	serializer_class = AudioPreprocessorSerializer
