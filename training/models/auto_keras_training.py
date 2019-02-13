from django.db import models

from training.models import AutoMlTraining


class AutoKerasTraining(AutoMlTraining):
	time_limit = models.IntegerField(null=True, blank=True)
	verbose = models.BooleanField(default=True)