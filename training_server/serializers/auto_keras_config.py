from rest_framework import serializers

from training_server.models import AutoKerasConfig


class AutoKerasConfigSerializer(serializers.ModelSerializer):
	class Meta:
		model = AutoKerasConfig
		exclude = ('training_time', 'framework')
