from evaluation.models.validator import Validator
from experiment_administration.models.experiment_supervisor import ExperimentSupervisor
from preprocessing.models.audio_preprocessor import AudioPreprocessor

# Preprocess audio files
from preprocessing.models.picture_preprocessor import PicturePreprocessor
from training.models import AutoSklearnTraining, AutoKerasTraining, TpotTraining, AutoMlTraining

def start_experiment(runtimes_seconds, experiment_id):
	error_log, created = ExperimentSupervisor.objects.get_or_create(name='Experiment log ' + str(experiment_id))

	print('in experiment')
	if created:
		error_log.step = 0
		error_log.model_ids = []
		error_log.save()
	else:
		# Skipping process where error happened
		if error_log.step == 1337:
			all_model_ids = error_log.model_ids
			# Remove First model_id to skip
			if len(all_model_ids)>0:
				error_log.model_ids.remove(all_model_ids[0])
		else:
			error_log.step += 1
			error_log.save()

	if error_log.step < 1:

		audio_preprocess_config = AudioPreprocessor.objects.create(
			transform_categorical_to_binary=True,
			binary_true_name='pc',
			input_folder_name='/wav/',
			input_data_type='wav',
			preprocessing_name='audio_preprocessing_experiment'
		)
		# audio_preprocess_config.transform_media_files_to_npy(is_audio=True)
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
			binary_true_name='pc',
			input_folder_name='/png/',
			input_data_type='png',
			preprocessing_name='picture_preprocessing_experiment'
		)
		pictures_preprocess_config.transform_media_files_to_npy(is_audio=False)
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

	for runtime_seconds in runtimes_seconds:
		loop_counter = runtimes_seconds.index(runtime_seconds)
		# ===================================================
		# Experiment series ask
		# ===================================================

		memory_limit = 10000

		if error_log.step < 3 + (12 * loop_counter):
			ask_config_1h_mc_audio = AutoSklearnTraining.objects.create(
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

			ask_config_1h_mc_audio.train()
			print('Training 1 success!')
			error_log.model_ids.append(ask_config_1h_mc_audio.id)
			error_log.step += 1
			error_log.save()

		if error_log.step < 4 + (12 * loop_counter):
			ask_config_1h_bc_audio = AutoSklearnTraining.objects.create(
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
			ask_config_1h_bc_audio.train()
			print('Training 2 success!')
			error_log.model_ids.append(ask_config_1h_bc_audio.id)
			error_log.step += 1
			error_log.save()

		if error_log.step < 5 + (12 * loop_counter):
			ask_config_1h_mc_png = AutoSklearnTraining.objects.create(
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
			ask_config_1h_mc_png.train()
			print('Training 3 success!')
			error_log.model_ids.append(ask_config_1h_mc_png.id)
			error_log.step += 1
			error_log.save()

		if error_log.step < 6 + (12 * loop_counter):
			ask_config_1h_bc_png = AutoSklearnTraining.objects.create(
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
			ask_config_1h_bc_png.train()
			print('Training 4 success!')
			error_log.model_ids.append(ask_config_1h_bc_png.id)
			error_log.step += 1
			error_log.save()

		# ===================================================
		# Experiment series AutoKeras
		# ===================================================

		if error_log.step < 7 + (12 * loop_counter):
			ak_config_1h_mc_audio = AutoKerasTraining.objects.create(
				time_limit=runtime_seconds,
				training_name='Auto Keras ' + str(runtime_seconds) + ' seconds multiclass classification with audio input',
				preprocessing_object=audio_files_preprocessed,
				load_files_from='preprocessing_job',
				task_type='multiclass_classification',
				framework='auto_keras',
				freeze_results=True,
				training_triggered=True
			)
			ak_config_1h_mc_audio.train()
			print('Training 5 success!')
			error_log.step += 1
			error_log.save()

		if error_log.step < 8 + (12 * loop_counter):
			ak_config_1h_bc_audio = AutoKerasTraining.objects.create(
				time_limit=runtime_seconds,
				training_name='Auto Keras ' + str(runtime_seconds) + ' seconds binary classification with audio input',
				preprocessing_object=audio_files_preprocessed,
				load_files_from='preprocessing_job',
				task_type='binary_classification',
				framework='auto_keras',
				freeze_results=True,
				training_triggered=True
			)
			ak_config_1h_bc_audio.train()
			print('Training 6 success!')
			error_log.model_ids.append(ak_config_1h_bc_audio.id)
			error_log.step += 1
			error_log.save()

		if error_log.step < 9 + (12 * loop_counter):
			ak_config_1h_mc_png = AutoKerasTraining.objects.create(
				time_limit=runtime_seconds,
				training_name='Auto Keras ' + str(runtime_seconds) + ' seconds multiclass classification with picture input',
				preprocessing_object=pictures_preprocessed,
				load_files_from='preprocessing_job',
				framework='auto_keras',
				task_type='multiclass_classification',
				freeze_results=True,
				training_triggered=True
			)
			ak_config_1h_mc_png.train()
			print('Training 7 success!')
			error_log.model_ids.append(ak_config_1h_mc_png.id)
			error_log.step += 1
			error_log.save()

		if error_log.step < 10 + (12 * loop_counter):
			ak_config_1h_bc_png = AutoKerasTraining.objects.create(
				time_limit=runtime_seconds,
				training_name='Auto Keras ' + str(runtime_seconds) + ' seconds binary classification with picture input',
				preprocessing_object=pictures_preprocessed,
				load_files_from='preprocessing_job',
				framework='auto_keras',
				task_type='binary_classification',
				freeze_results=True,
				training_triggered=True
			)
			ak_config_1h_bc_png.train()
			print('Training 8 success!')
			error_log.model_ids.append(ak_config_1h_bc_png.id)
			error_log.step += 1
			error_log.save()

		# ===================================================
		# Experiment series Tpot
		# ===================================================

		if error_log.step < 11 + (12 * loop_counter):
			tpot_training_1h_mc_audio = TpotTraining.objects.create(
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
			tpot_training_1h_mc_audio.train()
			print('Training 9 success!')
			error_log.model_ids.append(tpot_training_1h_mc_audio.id)
			error_log.step += 1
			error_log.save()

		if error_log.step < 12 + (12 * loop_counter):
			tpot_training_1h_bc_audio = TpotTraining.objects.create(
				verbosity=2,
				max_time_mins=int(runtime_seconds / 60),
				max_eval_time_mins=int(runtime_seconds/60/5),
				population_size=4,
				generations=3,
				training_name='Tpot ' + str(runtime_seconds) + ' seconds binary classification with audio input',
				framework='tpot',
				preprocessing_object=audio_files_preprocessed,
				load_files_from='preprocessing_job',
				task_type='binary_classification',
				freeze_results=True,
				training_triggered=True,
				cv=2
			)
			tpot_training_1h_bc_audio.train()
			print('Training 10 success!')
			error_log.model_ids.append(tpot_training_1h_bc_audio.id)
			error_log.step += 1
			error_log.save()

		if error_log.step < 13 + (12 * loop_counter):
			tpot_training_1h_mc_png = TpotTraining.objects.create(
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
			tpot_training_1h_mc_png.train()
			print('Training 11 success!')
			error_log.model_ids.append(tpot_training_1h_mc_png.id)
			error_log.step += 1
			error_log.save()

		if error_log.step < 14 + (12 * loop_counter):
			tpot_training_1h_bc_png = TpotTraining.objects.create(
				verbosity=2,
				max_time_mins=int(runtime_seconds / 60),
				max_eval_time_mins=int(runtime_seconds/60/5),
				population_size=4,
				generations=3,
				training_name='Tpot ' + str(runtime_seconds) + ' seconds binary classification with picture input',
				framework='tpot',
				preprocessing_object=pictures_preprocessed,
				load_files_from='preprocessing_job',
				task_type='binary_classification',
				freeze_results=True,
				training_triggered=True,
				cv=2
			)
			tpot_training_1h_bc_png.train()
			print('Training 12 success!')
			error_log.model_ids.append(tpot_training_1h_bc_png.id)
			error_log.step += 1
			error_log.save()

	for training_config_id in error_log.model_ids:
		error_log.step = 1337

		training_config = AutoMlTraining.objects.get(id=training_config_id)
		if training_config.model_path:
			ac = Validator.objects.create(
				model=training_config,
				scoring_strategy='accuracy'
			)
			ac.predict()
			if training_config.task_type=='binary_classification':
				pr = Validator.objects.create(
					model=training_config,
					scoring_strategy='precission'
				)
				pr.predict()
				ra = Validator.objects.create(
					model=training_config,
					scoring_strategy='roc_auc'
				)
				ra.predict()
		# TODO trigger calc of valr.
		error_log.model_ids.remove(training_config_id)
		error_log.save()
	error_log.step += 1

	return print('success!')
