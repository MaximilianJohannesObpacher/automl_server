

def make_categorical_binary(labels, true_name):
	bin_labels = []

	for label in labels:
		if label==true_name:
			bin_labels.append(1)
		else:
			bin_labels.append(0)

	return bin_labels