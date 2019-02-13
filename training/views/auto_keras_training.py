from rest_framework import viewsets

from training.models import AutoKerasTraining
from training.serializers.auto_keras_training import AutoKerasTrainingSerializer


class AutoKerasTrainingViewSet(viewsets.ModelViewSet):
	queryset = AutoKerasTraining.objects.all()
	serializer_class = AutoKerasTrainingSerializer