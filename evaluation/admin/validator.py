from django.contrib import admin
from django.utils.safestring import mark_safe

from evaluation.management.evaluate_accuracy import evaluate_all_models_accuracy
from evaluation.models.validator import Validator
from print_models import print_models


def evaluate_every_models_accuracy(modeladmin, request, queryset):
	evaluate_all_models_accuracy()
	evaluate_every_models_accuracy.short_description = "evaluate all accuracies!"

def print_all_models(modeladmin, request, queryset):
	print_models()
	print_all_models().short_description = "print_all_models!"

class ValidatorAdmin(admin.ModelAdmin):
	list_display = ('framework', 'model_short_characterisation', 'status', 'classification_task', 'scoring_strategy', 'score')
	fieldsets = (
		('Settings:', {'fields': ('model', 'scoring_strategy')}),
		('Results:', {'fields': ('status', 'score', 'additional_remarks', 'confusion_matrix', 'conf_matrix')})
	)
	readonly_fields = ('status', 'score', 'additional_remarks', 'conf_matrix')
	list_filter = ('status', 'scoring_strategy')
	actions = [evaluate_every_models_accuracy, print_all_models]


	def conf_matrix(self, obj):
		return mark_safe('<img src="{url}", style="width:70%" />'.format(
			url=obj.confusion_matrix.url,
		)
	)

	def framework(self, obj):
		return obj.model.framework

	framework.admin_order_field = 'framework'
	framework.short_description = 'AutoML-Framework'

	def classification_task(self, obj):
		return obj.model.task_type

	classification_task.admin_order_field = 'classification_task'
	classification_task.short_description = 'Classification Task'

	def model_short_characterisation(self, obj):
		return str(obj.model.framework) + '_' + str(obj.model.training_time)

	model_short_characterisation.admin_order_field = 'model_short_characterisation'
	model_short_characterisation.short_description = 'Model (Framework + Training Time)'

	def save_model(self, request, obj, form, change):
		obj.status = 'waiting'
		obj.save()
		obj.predict()


admin.site.register(Validator, ValidatorAdmin)
