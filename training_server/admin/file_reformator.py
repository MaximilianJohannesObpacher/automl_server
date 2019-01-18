from django.contrib import admin

from training_server.models import FileReformater


class FileReformatorAdmin(admin.ModelAdmin):

	def get_readonly_fields(self, request, obj=None):
		return ('status', 'model_path', 'date_trained', 'training_triggered', 'additional_remarks', 'training_time')

	def save_model(self, request, obj, form, change):
		pass  # Not saving the basemodel algorithmconfig, but instead the model autosklearn_config or tpot_config in the response add method

	# TODO refactor for one set
	# depending on framework selection forward to the submodel
	def response_add(self, request, obj, post_url_continue=None):

		if obj.input_file_format == 'parquet':
			print('Allow every format, time series')
		elif obj.input_file_format == 'wav':
			print('to numpy, classification or binary')
		elif obj.input_file_format == 'png':
			print('to numpy using klaidi, classification or binary')

		# TODO this is the fileformat changer
		# TODO Add additional task changer

		#   tpot_config = TpotConfig.objects.create(**conf_dict)
		#   redirect_path = '/admin/training_server/tpotconfig/' + str(tpot_config.id) + '/change/'
		# return HttpResponseRedirect(redirect_path)

	def has_change_permission(self, request, obj=None):
		return False

	def has_delete_permission(self, request, obj=None):
		return False


admin.site.register(FileReformater, FileReformatorAdmin) # TODO correct typo
