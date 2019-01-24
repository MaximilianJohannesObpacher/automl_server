from pathlib import Path

import cv2
import glob
import os

import librosa
import numpy
from PIL import Image

from automl_server.settings import AUTO_ML_DATA_PATH


def create_resized_copy_of_image(img, max_dimen, image_path):
	img = img.resize((max_dimen, max_dimen))

	image_path_components = image_path.split('/')
	folder_name = image_path_components[-2]

	# obtain the path to the image
	file_name = image_path_components.pop()

	image_root = image_path.split(folder_name)[0] + folder_name
	image_root = image_root.replace('png', 'png_resize')

	if not os.path.exists(image_root):
		path = Path(image_root)
		path.mkdir(parents=True, exist_ok=True)

	if os.path.exists(image_root + '/' + file_name):
		os.remove(image_root + '/' + file_name)

	img.save(image_root + '/' + file_name)


def resize_images():
	for filepath in glob.iglob(AUTO_ML_DATA_PATH + '/png/**/*.png', recursive=True):
		img = Image.open(filepath)
		image_path = img.filename
		create_resized_copy_of_image(img, 128, image_path)



def generate_image_array():
	loaded_features = numpy.array([ cv2.imread(fn) for fn in glob.iglob(AUTO_ML_DATA_PATH + '/png_resize/**/*.png', recursive=True)])
	labels = [fn.split('/')[-2] for fn in glob.iglob(AUTO_ML_DATA_PATH + '/png_resize/**/*.png', recursive=True)]

	for i in range(len(loaded_features)):
		# first order difference, computed over 9-step window
		loaded_features[i, :, :, 1] = librosa.feature.delta(loaded_features[i, :, :, 0])

		# for using 3 dimensional array to use ResNet and other frameworks
		loaded_features[i, :, :, 2] = librosa.feature.delta(loaded_features[i, :, :, 1])

	loaded_features = numpy.transpose(loaded_features, (0, 2, 1, 3))

	return loaded_features, labels
