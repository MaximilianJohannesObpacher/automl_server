from django.db import models

from training_server.models import AlgorithmConfig


class AutoKerasConfig(AlgorithmConfig):
	time_limit = models.IntegerField(null=True, blank=True)
	verbose = models.BooleanField(default=True)