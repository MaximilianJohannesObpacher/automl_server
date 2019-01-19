import os
from shutil import copyfile

from automl_server.settings import AUTO_ML_DATA_PATH


def png_wav_folder_sorter():
	for subfolders in os.walk(AUTO_ML_DATA_PATH + '/unsorted'):
		for subfolder in subfolders[1]:
			for files in os.walk(AUTO_ML_DATA_PATH + '/unsorted/' + subfolder):
				for file in files[2]:
					for file_type in ['wav', 'png']:
						if file.split('.').pop() == file_type:
							folder = subfolder.split('/').pop()
							if not os.path.exists(AUTO_ML_DATA_PATH + '/'+ file_type +'/' + folder):
								os.makedirs(AUTO_ML_DATA_PATH + '/'+ file_type +'/' + folder)
							copyfile(files[0] + '/' + file, AUTO_ML_DATA_PATH + '/'+ file_type +'/' + folder + '/' + file)