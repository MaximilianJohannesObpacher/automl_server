from django.db import models

from training.models import AutoMlTraining


class ValidationResult(models.Model):
	IN_PROGRESS = 'in_progress'
	SUCCESS = 'success'
	FAIL = 'fail'
	WAITING = 'waiting'

	STATUS_CHOICES = (
		(WAITING, 'Waiting for thread'),
		(IN_PROGRESS, 'In progress'),
		(SUCCESS, 'Success'),
		(FAIL, 'Fail')
	)

	ACCURACY = 'accuracy'
	PRECISION ='precision'
	ROC_AUC = 'roc_auc'

	#TODO
	SCORING_CHOICES = (
		(ACCURACY, 'Accuracy'),
		(PRECISION, 'Precision'),
		(ROC_AUC, 'Roc_Auc')
	)

	scoring_strategy = models.CharField(max_length=30, choices=SCORING_CHOICES, blank=True, null=True)
	score = models.FloatField(max_length=10, blank=True, null=True)
	additional_remarks = models.CharField(max_length=2048, blank=True, null=True)
	status = models.CharField(max_length=32, choices=STATUS_CHOICES, blank=True, null=True)
	model = models.ForeignKey(AutoMlTraining, null=True, blank=True)

