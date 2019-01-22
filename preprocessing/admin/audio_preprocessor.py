from django.contrib import admin

from preprocessing.file_preprocessing.audio_to_npy import transform_all_audio_files_to_npy
from preprocessing.models.audio_preprocessor import AudioPreprocessor


class AudioPreprocessorAdmin(admin.ModelAdmin):
    list_display = ('status','input_folder_name')
    list_filter = ('status',)
    readonly_fields = ('status', 'additional_remarks', 'training_features_path', 'training_labels_path','evaluation_features_path', 'evaluation_labels_path', 'training_labels_path_binary', 'evaluation_labels_path_binary')




    def save_model(self, request, obj, form, change):
        obj = transform_all_audio_files_to_npy(obj) # TODO Find out how to make async
        obj.save()


admin.site.register(AudioPreprocessor, AudioPreprocessorAdmin)
