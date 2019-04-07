from rest_framework import serializers

from training.models import TpotTraining


class TpotTrainingSerializer(serializers.ModelSerializer):
	class Meta:
		model = TpotTraining
		exclude = ('training_time', 'framework')
