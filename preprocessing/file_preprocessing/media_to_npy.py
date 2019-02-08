import cv2
import datetime
import glob
import os

import numpy
from scipy.io.wavfile import read



def pad_trunc_seq_rewrite(x, max_len):
	if x.shape[1] < max_len:
		pad_shape = (x.shape[0], max_len - x.shape[1])
		pad = numpy.ones(pad_shape) * numpy.log(1e-8)
		x_new = numpy.hstack((x, pad))

	# no pad necessary - truncate
	else:
		x_new = x[:, 0:max_len]
	return x_new
