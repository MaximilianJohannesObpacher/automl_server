from django.contrib.postgres.fields import ArrayField
from django.db import models

class ErrorLog(models.Model):
	name = models.CharField(max_length=1024)
	step = models.IntegerField(null=True, blank=True)
	model_ids = ArrayField(default=[], blank=True, base_field=models.IntegerField(null=True, blank=True), size=50 )