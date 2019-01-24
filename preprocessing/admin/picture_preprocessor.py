from preprocessing.admin import AudioPreprocessorAdmin, admin
from preprocessing.file_preprocessing.audio_to_npy import transform_all_audio_files_to_npy
from preprocessing.models.picture_preprocessor import PicturePreprocessor


class PicturePreprocessorAdmin(AudioPreprocessorAdmin):
	readonly_fields = (
	'status', 'additional_remarks', 'training_features_path', 'training_labels_path', 'evaluation_features_path',
	'evaluation_labels_path', 'training_labels_path_binary', 'evaluation_labels_path_binary', 'bands', 'frames')

	def save_model(self, request, obj, form, change):
		print(str(obj))
		obj = transform_all_audio_files_to_npy(obj, False)  # TODO Find out how to make async
		obj.input_data_type = 'png'
		obj.save()


admin.site.register(PicturePreprocessor, PicturePreprocessorAdmin)
