from rest_framework import viewsets

from training_server.models import TpotConfig
from training_server.serializers.tpot_config import TpotConfigSerializer


class TpotConfigViewSet(viewsets.ModelViewSet):
	queryset = TpotConfig.objects.all()
	serializer_class = TpotConfigSerializer
