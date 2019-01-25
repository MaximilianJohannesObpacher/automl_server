from preprocessing.file_preprocessing.audio_picture_to_npy import transform_media_files_to_npy
from preprocessing.models.audio_preprocessor import AudioPreprocessor

# Preprocess audio files
from preprocessing.models.picture_preprocessor import PicturePreprocessor
from training_server.models import AutoSklearnConfig, AutoKerasConfig, TpotConfig, ErrorLog

from automl_systems.auto_sklearn.run import train as train_auto_sklearn
from automl_systems.auto_keras.run import train as train_auto_keras
from automl_systems.tpot.run import train as train_tpot


def start_experiment(runtime_seconds, experiment_id):
	error_log, created = ErrorLog.objects.get_or_create(name='Experiment log' + str(experiment_id))

	print('in experiment')
	if created:
		error_log.step = 0
	if not created:
		# Skipping process where error happened
		error_log.step += 1
		error_log.save()

	if error_log.step < 1:

		audio_preprocess_config = AudioPreprocessor.objects.create(
			transform_categorical_to_binary=True,
			binary_true_name='no_fat_behavior',
			input_folder_name='/wav/',
			input_data_type='wav',
			preprocessing_name='audio_preprocessing_experiment'
		)
		transform_media_files_to_npy(audio_preprocess_config, is_audio=True)
		audio_files_preprocessed = AudioPreprocessor.objects.get(id=audio_preprocess_config.id)
		print('Audio preprocess_success!')
		error_log.step += 1
		error_log.save()
	else:
		try:
			audio_files_preprocessed = AudioPreprocessor.objects.filter(
				preprocessing_name='audio_preprocessing_experiment'
			).last()
		except AudioPreprocessor.DoesNotExist:
			audio_files_preprocessed = None

	if error_log.step < 2:
		pictures_preprocess_config = PicturePreprocessor.objects.create(
			transform_categorical_to_binary=True,
			binary_true_name='no_fat_behavior',
			input_folder_name='/png/',
			input_data_type='png',
			preprocessing_name='picture_preprocessing_experiment'
		)
		transform_media_files_to_npy(pictures_preprocess_config, is_audio=False)
		pictures_preprocessed = PicturePreprocessor.objects.get(id=pictures_preprocess_config.id)
		print('Picture preprocess_success!')

		error_log.step += 1
		error_log.save()

	else:
		try:
			pictures_preprocessed = PicturePreprocessor.objects.filter(
				preprocessing_name='picture_preprocessing_experiment'
			).last()
		except PicturePreprocessor.DoesNotExist:
			pictures_preprocessed = None

	# ===================================================
	# Experiment series ask
	# ===================================================

	memory_limit = 8192

	if error_log.step < 3:
		ask_config_1h_mc_audio = AutoSklearnConfig.objects.create(
			run_time=runtime_seconds,
			per_instance_runtime=int(runtime_seconds/10),
			memory_limit=memory_limit,
			training_name='Auto Sklearn ' + str(runtime_seconds) + ' seconds multiclass classification with audio input',
			preprocessing_object=audio_files_preprocessed,
			framework='auto_sklearn',
			load_files_from='preprocessing_job',
			task_type='multiclass_classification',
			freeze_results=True,
			training_triggered=True
		)

		train_auto_sklearn(str(ask_config_1h_mc_audio.id))
		print('Training 1 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 4:
		ask_config_1h_bc_audio = AutoSklearnConfig.objects.create(
			run_time=runtime_seconds,
			memory_limit=memory_limit,
			per_instance_runtime=int(runtime_seconds / 10),
			training_name='Auto Sklearn ' + str(runtime_seconds) + ' seconds binary classification with audio input',
			preprocessing_object=audio_files_preprocessed,
			framework='auto_sklearn',
			load_files_from='preprocessing_job',
			task_type='binary_classification',
			freeze_results=True,
			training_triggered=True
		)
		train_auto_sklearn(str(ask_config_1h_bc_audio.id))
		print('Training 2 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 5:
		ask_config_1h_mc_png = AutoSklearnConfig.objects.create(
			run_time=runtime_seconds,
			memory_limit=memory_limit,
			per_instance_runtime=int(runtime_seconds / 10),
			training_name='Auto Sklearn ' + str(runtime_seconds) + ' seconds multiclass classification with picture input',
			preprocessing_object=pictures_preprocessed,
			framework='auto_sklearn',
			load_files_from='preprocessing_job',
			task_type='multiclass_classification',
			freeze_results=True,
			training_triggered=True
		)
		train_auto_sklearn(str(ask_config_1h_mc_png.id))
		print('Training 3 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 6:
		ask_config_1h_bc_png = AutoSklearnConfig.objects.create(
			run_time=runtime_seconds,
			memory_limit=memory_limit,
			per_instance_runtime=int(runtime_seconds / 10),
			training_name='Auto Sklearn ' + str(runtime_seconds) + ' seconds binary classification with picture input',
			preprocessing_object=pictures_preprocessed,
			load_files_from='preprocessing_job',
			framework='auto_sklearn',
			task_type='binary_classification',
			freeze_results=True,
			training_triggered=True
		)
		train_auto_sklearn(str(ask_config_1h_bc_png.id))
		print('Training 4 success!')
		error_log.step += 1
		error_log.save()

	# ===================================================
	# Experiment series AutoKeras
	# ===================================================

	if error_log.step < 7:
		ak_config_1h_mc_audio = AutoKerasConfig.objects.create(
			time_limit=runtime_seconds,
			training_name='Auto Keras ' + str(runtime_seconds) + ' seconds multiclass classification with audio input',
			preprocessing_object=audio_files_preprocessed,
			load_files_from='preprocessing_job',
			task_type='multiclass_classification',
			framework='auto_keras',
			freeze_results=True,
			training_triggered=True
		)
		train_auto_keras(str(ak_config_1h_mc_audio.id))
		print('Training 5 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 8:
		ak_config_1h_bc_audio = AutoKerasConfig.objects.create(
			time_limit=runtime_seconds,
			training_name='Auto Keras ' + str(runtime_seconds) + ' seconds multiclass classification with audio input',
			preprocessing_object=audio_files_preprocessed,
			load_files_from='preprocessing_job',
			task_type='binary_classification',
			framework='auto_keras',
			freeze_results=True,
			training_triggered=True
		)
		train_auto_keras(str(ak_config_1h_bc_audio.id))
		print('Training 6 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 9:
		ak_config_1h_mc_png = AutoKerasConfig.objects.create(
			time_limit=runtime_seconds,
			training_name='Auto Keras ' + str(runtime_seconds) + ' seconds multiclass classification with picture input',
			preprocessing_object=pictures_preprocessed,
			load_files_from='preprocessing_job',
			framework='auto_keras',
			task_type='multiclass_classification',
			freeze_results=True,
			training_triggered=True
		)
		train_auto_keras(str(ak_config_1h_mc_png.id))
		print('Training 7 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 10:
		ak_config_1h_bc_png = AutoKerasConfig.objects.create(
			time_limit=runtime_seconds,
			training_name='Auto Keras ' + str(runtime_seconds) + ' seconds multiclass classification with picture input',
			preprocessing_object=pictures_preprocessed,
			load_files_from='preprocessing_job',
			framework='auto_keras',
			task_type='binary_classification',
			freeze_results=True,
			training_triggered=True
		)
		train_auto_keras(str(ak_config_1h_bc_png.id))
		print('Training 8 success!')
		error_log.step += 1
		error_log.save()

	# ===================================================
	# Experiment series Tpot
	# ===================================================

	if error_log.step < 11:
		tpot_config_1h_mc_audio = TpotConfig.objects.create(
			verbosity=2,
			max_time_mins=int(runtime_seconds / 60),
			max_eval_time_mins=int(runtime_seconds/60/5),
			population_size=4,
			generations=3,
			config_dict='TPOT light',
			training_name='Tpot ' + str(runtime_seconds) + ' seconds multiclass classification with audio input',
			preprocessing_object=audio_files_preprocessed,
			load_files_from='preprocessing_job',
			framework='tpot',
			task_type='multiclass_classification',
			freeze_results=True,
			training_triggered=True,
			cv=2
		)
		train_tpot(str(tpot_config_1h_mc_audio.id))
		print('Training 9 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 12:
		tpot_config_1h_bc_audio = TpotConfig.objects.create(
			verbosity=2,
			max_time_mins=int(runtime_seconds / 60),
			max_eval_time_mins=int(runtime_seconds/60/5),
			population_size=4,
			generations=3,
			training_name='Tpot ' + str(runtime_seconds) + ' seconds multiclass classification with audio input',
			framework='tpot',
			preprocessing_object=audio_files_preprocessed,
			load_files_from='preprocessing_job',
			task_type='binary_classification',
			freeze_results=True,
			training_triggered=True,
			cv=2
		)
		train_tpot(str(tpot_config_1h_bc_audio.id))
		print('Training 10 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 13:
		tpot_config_1h_mc_png = TpotConfig.objects.create(
			verbosity=2,
			max_time_mins=int(runtime_seconds / 60),
			max_eval_time_mins=int(runtime_seconds/60/5),
			population_size=4,
			generations=3,
			training_name='Tpot ' + str(runtime_seconds) + ' seconds multiclass classification with picture input',
			framework='tpot',
			preprocessing_object=pictures_preprocessed,
			load_files_from='preprocessing_job',
			task_type='multiclass_classification',
			freeze_results=True,
			training_triggered=True,
			cv=2
		)
		train_tpot(str(tpot_config_1h_mc_png.id))
		print('Training 11 success!')
		error_log.step += 1
		error_log.save()

	if error_log.step < 14:
		tpot_config_1h_bc_png = TpotConfig.objects.create(
			verbosity=2,
			max_time_mins=int(runtime_seconds / 60),
			max_eval_time_mins=int(runtime_seconds/60/5),
			population_size=4,
			generations=3,
			training_name='Tpot ' + str(runtime_seconds) + ' seconds multiclass classification with picture input',
			framework='tpot',
			preprocessing_object=pictures_preprocessed,
			load_files_from='preprocessing_job',
			task_type='binary_classification',
			freeze_results=True,
			training_triggered=True,
			cv=2
		)
		train_tpot(str(tpot_config_1h_bc_png.id))
		print('Training 12 success!')
		error_log.step += 1
		error_log.save()

	return print('success!')
