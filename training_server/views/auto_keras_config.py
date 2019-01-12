from rest_framework import viewsets

from training_server.models import AutoKerasConfig
from training_server.serializers.auto_keras_config import AutoKerasConfigSerializer


class AutoKerasConfigViewSet(viewsets.ModelViewSet):
	queryset = AutoKerasConfig.objects.all()
	serializer_class = AutoKerasConfigSerializer