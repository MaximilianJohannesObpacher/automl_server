from django.contrib import admin

from prediction.models.predictor import Predictor


class PredictorAdmin(admin.ModelAdmin):
	readonly_fields = ('result',)
	list_display = ('training', 'filename', 'result')

	def save_model(self, request, obj, form, change):
		obj.status = 'waiting'
		obj.save()
		obj.predict()

	def filename(self, obj):
		try:
			return obj.file.name.split('/')[1:][0]
		except:
			return None

	filename.admin_order_field = 'filename'
	filename.short_description = 'File Name'

admin.site.register(Predictor, PredictorAdmin)
