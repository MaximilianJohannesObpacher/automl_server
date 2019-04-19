import os

import numpy
from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from automl_server.settings import AUTO_ML_DATA_PATH
from preprocessing.models.file_preprocessor import FilePreprocessor
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

		# getting preprocessing file
		machine_preprocessor = FilePreprocessor.objects.filter(machine_id=akt.machine_id).last()
		if not machine_preprocessor:
			raise ValidationError("either set input data paths or link to a machine number that has a preprocessing job!")

		if akt.run_time == None and akt.per_instance_runtime == None:
			akt = AutoSklearnTraining.objects.create(run_time=3600, per_instance_runtime=360, load_files_from='preprocessing_job', preprocessing_object=machine_preprocessor, task_type='multiclass_classification', framework='auto-sklearn')
			akt = akt.save_labels(akt)
			akt = akt.config_algorithm(akt)
		else:
			akt = akt.save_labels(akt)

		akt.training_triggered = True
		akt.status = 'waiting'
		akt.save()
		print('runtime!' + str(akt.run_time))
		akt.train()
		return akt
