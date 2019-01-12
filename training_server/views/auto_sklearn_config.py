from rest_framework import viewsets, mixins

from training_server.models import AutoSklearnConfig
from training_server.serializers.auto_keras_config import AutoKerasConfigSerializer
from training_server.serializers.auto_sklearn_config import AutoSklearnConfigSerializer


class AutoSklearnConfigViewSet(viewsets.ModelViewSet):
	queryset = AutoSklearnConfig.objects.all()
	serializer_class = AutoSklearnConfigSerializer