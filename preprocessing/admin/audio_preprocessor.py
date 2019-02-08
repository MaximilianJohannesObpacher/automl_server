import os
from os import read

import numpy
from django.contrib import admin
from preprocessing.models.audio_preprocessor import AudioPreprocessor
from preprocessing.models.file_preprocessor import FilePreprocessor


class AudioPreprocessorAdmin(admin.ModelAdmin):
    list_display = ('status','input_folder_name')
    list_filter = ('status',)
    readonly_fields = ('status', 'additional_remarks', 'training_features_path', 'training_labels_path','evaluation_features_path', 'evaluation_labels_path', 'training_labels_path_binary', 'evaluation_labels_path_binary')
    exclude = ('input_data_type', 'input_folder_name', 'file_format')

    def save_model(self, request, obj, form, change):
        obj.input_folder_name='/wav/'
        FilePreprocessor.transform_media_files_to_npy(obj, True) # TODO Find out how to make async
        obj.input_data_type = 'wav'
        obj.save()


admin.site.register(AudioPreprocessor, AudioPreprocessorAdmin)
