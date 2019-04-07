from rest_framework import serializers

from training.models import AutoSklearnTraining


class AutoSklearnTrainingSerializer(serializers.ModelSerializer):
	class Meta:
		model = AutoSklearnTraining
		exclude = ('training_time', 'framework')

	include_estimators = serializers.ListField(child=serializers.CharField())
	exclude_estimators = serializers.ListField(child=serializers.CharField())
	include_preprocessors = serializers.ListField(child=serializers.CharField())
	exclude_preprocessors = serializers.ListField(child=serializers.CharField())
