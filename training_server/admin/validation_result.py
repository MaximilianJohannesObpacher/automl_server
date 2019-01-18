from django.contrib import admin

from automl_systems.predict import predict
from training_server.models.validation_result import ValidationResult


class ValidationResultAdmin(admin.ModelAdmin):
	list_display = ('framework', 'model_short_characterisation', 'status', 'classification_task', 'scoring_strategy', 'score')
	fieldsets = (
		('Settings:', {'fields': ('model', 'scoring_strategy')}),
		('Results:', {'fields': ('status', 'score', 'additional_remarks',)})
	)
	readonly_fields = ('status', 'score', 'additional_remarks')
	list_filter = ('status', 'scoring_strategy')

	def framework(self, obj):
		return obj.model.framework

	framework.admin_order_field = 'framework'
	framework.short_description = 'AutoML-Framework'

	def classification_task(self, obj):
		if obj.model.make_one_hot_encoding_task_binary:
			return 'binary'
		else:
			return 'multiclass'

	classification_task.admin_order_field = 'classification_task'
	classification_task.short_description = 'Classification Task'

	def model_short_characterisation(self, obj):
		return str(obj.model.framework) + '_' + str(obj.model.training_time)

	model_short_characterisation.admin_order_field = 'model_short_characterisation'
	model_short_characterisation.short_description = 'Model (Framework + Training Time)'

	def save_model(self, request, obj, form, change):
		obj.status = 'waiting'
		obj.save()
		predict(obj)



admin.site.register(ValidationResult, ValidationResultAdmin)