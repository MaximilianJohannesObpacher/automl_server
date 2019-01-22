from django.db import models


class FilePreprocessorManager(models.Manager):
	def get_queryset(self):
		return super().get_queryset().filter(status='success')


class FilePreprocessor(models.Model):
	objects = FilePreprocessorManager()

	SUCCESS = 'success'
	FAIL = 'fail'

	STATUS_CHOICES = (
		(SUCCESS, 'Success'),
		(FAIL, 'Fail')
	)

	NUMPY_ARRAY = 'npy'
	CSV = 'csv'
	PICKLE = 'pkl'

	OUTPUT_CHOICES = (
		(NUMPY_ARRAY, 'npy'),
		(CSV, 'csv'),
		(PICKLE, 'pkl')
	)
	input_folder_name = models.CharField(max_length=256, default='/wav/', blank=True, null=True)

	status = models.CharField(choices=STATUS_CHOICES, max_length=32, help_text='status of the training', null=True,
	                          blank=True)
	additional_remarks = models.CharField(null=True, blank=True, max_length=2048,
                                      help_text='Additional Information about the training. E.g. Information about failed trainings are logged here in case a training fails!')

	training_features_path = models.CharField(max_length=256, null=True, blank=True)
	training_labels_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_features_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_labels_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_labels_path_binary = models.CharField(max_length=256, null=True, blank=True)
	training_labels_path_binary = models.CharField(max_length=256, null=True, blank=True)

	def __str__(self):
		return str(self.input_folder_name) + '_' + str(self.training_features_path)