from django.core.management import BaseCommand

from automl_systems.predict import predict
from evaluation.models.validation_result import ValidationResult
from training.models import AutoMlTraining

class Command(BaseCommand):
	help = 'Starts an evaluation!'

	def handle(self, *args, **options):
		print('about to start eval!')
		evaluate_all_models_accuracy()
		print('finished eval!')

def evaluate_all_models_accuracy():
	i=1
	for ac in AutoMlTraining.objects.all():
		print(i)
		if ac.model_path:
			vr = ValidationResult.objects.create(
				model=ac,
				scoring_strategy='accuracy'
			)
			predict(vr)
		i+=1