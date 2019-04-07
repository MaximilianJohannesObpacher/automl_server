from rest_framework import viewsets

from training.models import TpotTraining
from training.serializers.tpot_training import TpotTrainingSerializer


class TpotTrainingViewSet(viewsets.ModelViewSet):
	queryset = TpotTraining.objects.all()
	serializer_class = TpotTrainingSerializer
