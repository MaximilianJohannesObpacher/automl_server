from rest_framework import viewsets

from preprocessing.models.picture_preprocessor import PicturePreprocessor
from preprocessing.serializers.picture_preprocessor import PicturePreprocessorSerializer


class PicturePreprocessorViewSet(viewsets.ModelViewSet):
	queryset = PicturePreprocessor.objects.all()
	serializer_class = PicturePreprocessorSerializer