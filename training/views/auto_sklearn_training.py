from rest_framework import viewsets, mixins

from training.models import AutoSklearnTraining
from training.serializers.auto_sklearn_training import AutoSklearnTrainingSerializer


class AutoSklearnTrainingViewSet(viewsets.ModelViewSet):
	queryset = AutoSklearnTraining.objects.all()
	serializer_class = AutoSklearnTrainingSerializer