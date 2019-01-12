from rest_framework import serializers

from training_server.models import TpotConfig


class TpotConfigSerializer(serializers.ModelSerializer):
	class Meta:
		model = TpotConfig
		exclude = ('training_time', 'framework')
