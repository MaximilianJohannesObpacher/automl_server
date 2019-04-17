from rest_framework import serializers

from evaluation.models.validator import Validator


class ValidatorSerializer(serializers.ModelSerializer):
	class Meta:
		model = Validator
		fields = '__all__'

	def create(self, validated_data):
		obj = Validator.objects.create(**validated_data)
		obj.predict()
		return obj
	