import glob
import os
import pickle

from automl_server.settings import AUTO_ML_DATA_PATH_EXP


def print_models():
	for model_path in os.listdir(AUTO_ML_DATA_PATH_EXP):
		if not model_path == '.DS_Store':
			print(len(os.listdir(AUTO_ML_DATA_PATH_EXP)))
			print('Batz,' + model_path)
			bytes_in = bytearray(0)
			input_size = os.path.getsize(AUTO_ML_DATA_PATH_EXP + '/' + model_path)
			max_bytes = 2 ** 31 - 1
			with open(AUTO_ML_DATA_PATH_EXP + '/' + model_path, 'rb') as f:
				for _ in range(0, input_size, max_bytes):
					bytes_in += f.read(max_bytes)
			my_model = pickle.loads(bytes_in)
			f = open(model_path + ".txt", "w+")
			f.write('Model Name: ' + model_path + ' , Config: '+ str(my_model.show_models()))
			f.close()
			print(my_model.show_models())