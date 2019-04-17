from rest_framework import serializers

from prediction.models.predictor import Predictor


class PredictorSerializer(serializers.ModelSerializer):
	class Meta:
		model = Predictor
		fields = '__all__'

	def create(self, validated_data):
		pred = Predictor.objects.create(**validated_data)
		pred.predict()
		return pred
