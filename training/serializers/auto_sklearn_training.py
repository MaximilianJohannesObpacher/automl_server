from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from training.models import AutoSklearnTraining


class AutoSklearnTrainingSerializer(serializers.ModelSerializer):
	class Meta:
		model = AutoSklearnTraining
		exclude = ('training_time', 'framework')

	include_estimators = serializers.ListField(child=serializers.CharField())
	exclude_estimators = serializers.ListField(child=serializers.CharField())
	include_preprocessors = serializers.ListField(child=serializers.CharField())
	exclude_preprocessors = serializers.ListField(child=serializers.CharField())

	def create(self, validated_data):
		akt = AutoSklearnTraining.objects.create(**validated_data)
		if akt.load_files_from == 'preprocessing_job':
			if not akt.preprocessing_object:
				raise ValidationError('No preprocessing object selected!')
			akt.training_data_filename = akt.preprocessing_object.training_features_path
			akt.validation_data_filename = akt.preprocessing_object.evaluation_features_path
			if akt.task_type == 'binary_classification':
				akt.training_labels_filename = akt.preprocessing_object.training_labels_path_binary
				akt.validation_labels_filename = akt.preprocessing_object.evaluation_labels_path_binary
			else:
				akt.training_labels_filename = akt.preprocessing_object.training_labels_path
				akt.validation_labels_filename = akt.preprocessing_object.evaluation_labels_path
		print(str(akt.training_data_filename) + str(akt.validation_data_filename) + str(
			akt.training_labels_filename) + str(akt.validation_labels_filename))
		akt.training_triggered = True
		akt.status = 'waiting'
		akt.save()
		akt.train()
		return akt