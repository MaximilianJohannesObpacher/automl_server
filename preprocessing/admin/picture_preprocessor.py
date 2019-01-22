from preprocessing.admin import AudioPreprocessorAdmin, admin
from preprocessing.file_preprocessing.audio_to_npy import transform_all_audio_files_to_npy
from preprocessing.models.picture_preprocessor import PicturePreprocessor


class PicturePreprocessorAdmin(AudioPreprocessorAdmin):

	def save_model(self, request, obj, form, change):
		print(str(obj))
		obj = transform_all_audio_files_to_npy(obj, False)  # TODO Find out how to make async
		obj.save()


admin.site.register(PicturePreprocessor, PicturePreprocessorAdmin)
