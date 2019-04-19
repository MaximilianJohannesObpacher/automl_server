from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from preprocessing.models.file_preprocessor import FilePreprocessor
from training.models import TpotTraining


class TpotTrainingSerializer(serializers.ModelSerializer):
	class Meta:
		model = TpotTraining
		exclude = ('training_time', 'framework')

	def create(self, validated_data):
		akt = TpotTraining.objects.create(**validated_data)

		# getting preprocessing file
		machine_preprocessor = FilePreprocessor.objects.filter(machine_id=akt.machine_id).last()
		if not machine_preprocessor:
			raise ValidationError(
				"either set input data paths or link to a machine number that has a preprocessing job!")

		if akt.max_time_mins == None and akt.max_eval_time_mins == None:
			akt = TpotTraining.objects.create(max_time_mins=60, max_eval_time_mins=5,
			                                         load_files_from='preprocessing_job',
			                                         preprocessing_object=machine_preprocessor,
			                                         task_type='multiclass_classification', framework='tpot')
			akt = akt.save_labels(akt)
			akt = akt.config_algorithm(akt)
		else:
			akt = akt.save_labels(akt)

		akt.training_triggered = True
		akt.status = 'waiting'
		akt.save()
		akt.train()
		return akt