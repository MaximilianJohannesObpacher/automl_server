from rest_framework import serializers

from training.models import AutoKerasTraining


class AutoKerasTrainingSerializer(serializers.ModelSerializer):
	class Meta:
		model = AutoKerasTraining
		exclude = ('training_time', 'framework')
