from rest_framework import serializers

from prediction.models.predictor import Predictor


class PredictorSerializer(serializers.ModelSerializer):
	class Meta:
		model = Predictor
		fields = '__all__'