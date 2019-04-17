from django.http import HttpResponse
from requests import Response
from rest_framework import viewsets

from prediction.models.predictor import Predictor
from prediction.serializers.predictor import PredictorSerializer


class PredictorViewSet(viewsets.ModelViewSet):
	queryset = Predictor.objects.all()
	serializer_class = PredictorSerializer

