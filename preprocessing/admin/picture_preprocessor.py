from preprocessing.admin import AudioPreprocessorAdmin, admin
from preprocessing.models.picture_preprocessor import PicturePreprocessor


class PicturePreprocessorAdmin(AudioPreprocessorAdmin):
	readonly_fields = (
	'status', 'additional_remarks', 'training_features_path', 'training_labels_path', 'evaluation_features_path',
	'evaluation_labels_path', 'training_labels_path_binary', 'evaluation_labels_path_binary')

	def save_model(self, request, obj, form, change):
		obj.input_folder_name = '/png/'
		obj = obj.transform_media_files_to_npy(False)  # TODO Find out how to make async
		obj.input_data_type = 'png'
		obj.save()


admin.site.register(PicturePreprocessor, PicturePreprocessorAdmin)
