from django.contrib import admin

from automl_systems.predict import predict
from training_server.models.validation_result import ValidationResult


class ValidationResultAdmin(admin.ModelAdmin):
	list_display = ('status', 'scoring_strategy', 'score')

	def save_model(self, request, obj, form, change):
		obj.status = 'waiting'
		obj.save()
		predict(obj)


admin.site.register(ValidationResult, ValidationResultAdmin)