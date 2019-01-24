from django.db import models

class ErrorLog(models.Model):
	name = models.CharField(max_length=1024)
	step = models.IntegerField(max_length=1024, null=True, blank=True)