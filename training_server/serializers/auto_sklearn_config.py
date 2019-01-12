from rest_framework import serializers

from training_server.models import AutoSklearnConfig


class AutoSklearnConfigSerializer(serializers.ModelSerializer):
	class Meta:
		model = AutoSklearnConfig
		exclude = ('training_time', 'framework')

	include_estimators = serializers.ListField(child=serializers.CharField())
	exclude_estimators = serializers.ListField(child=serializers.CharField())
	include_preprocessors = serializers.ListField(child=serializers.CharField())
	exclude_preprocessors = serializers.ListField(child=serializers.CharField())
