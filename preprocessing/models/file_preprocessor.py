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

	PNG = 'png'
	WAV = 'wav'

	datatype_choices = (
		(PNG, 'png'),
		(WAV, 'wav')
	)

	status = models.CharField(choices=STATUS_CHOICES, max_length=32, help_text='status of the training', null=True,
	                          blank=True)
	additional_remarks = models.CharField(null=True, blank=True, max_length=2048,
                                      help_text='Additional Information about the training. E.g. Information about failed trainings are logged here in case a training fails!')
	transform_categorical_to_binary = models.BooleanField(default=False, help_text='should the data be labeled binary as well?')
	training_features_path = models.CharField(max_length=256, null=True, blank=True)
	training_labels_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_features_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_labels_path = models.CharField(max_length=256, null=True, blank=True)
	evaluation_labels_path_binary = models.CharField(max_length=256, null=True, blank=True)
	training_labels_path_binary = models.CharField(max_length=256, null=True, blank=True)
	binary_true_name = models.CharField(max_length=256, null=True, blank=True, default='perfect_condition', help_text='if binary transform categorical data to binary is true, all files in folder labeled with this name will be labeled as True while all other data will be labeled as false.')
	input_folder_name = models.CharField(max_length=256, default='', blank=True, null=True)
	input_data_type = models.CharField(blank=True, null=True, choices=datatype_choices, max_length=32)
	preprocessing_name = models.CharField(max_length=255, null=True, blank=True)

	def __str__(self):
		return str(self.input_folder_name) + '_' + str(self.training_features_path)