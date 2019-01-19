import os

import pandas as pd

from automl_server.settings import AUTO_ML_DATA_PATH


def parquet_to_csv(filename):
	df = pd.read_parquet((os.path.join(AUTO_ML_DATA_PATH + '/parquet/', filename)), engine='pyarrow')
	filename = filename.replace('.parquet', '.csv')
	df.to_csv(os.path.join(AUTO_ML_DATA_PATH + '/csv/', filename))

def preprocess_csv(filename):
	if os.path.exists(os.path.join(AUTO_ML_DATA_PATH + '/pickle/features_' + filename)) and os.path.exists(os.path.join(AUTO_ML_DATA_PATH + '/pickle/target_slave_' + filename)) and os.path.exists(os.path.join(AUTO_ML_DATA_PATH + '/pickle/target_ak0_' + filename)):
		return pd.read_pickle(os.path.join(AUTO_ML_DATA_PATH + '/pickle/features_' + filename)), pd.read_pickle(os.path.join(AUTO_ML_DATA_PATH + '/pickle/target_slave_' + filename)), pd.read_pickle(os.path.join(AUTO_ML_DATA_PATH + '/pickle/target_ak0_' + filename))
	else:
		df = pd.read_csv(os.path.join(AUTO_ML_DATA_PATH + '/csv/', filename))
		df['timestamp'] = pd.to_datetime(df.timestamp)
		df['timestamp_date'] = df['timestamp'].dt.day + df['timestamp'].dt.month*31 + df['timestamp'].dt.year * 365
		# board_remaining_time create 11 time series
		features = df.groupby(['cpuType', 'cpuBoard', 'remainingTime']).size().reset_index()
		target_slave = df.groupby(['cpuType', 'cpuBoard', 'remainingTime'], as_index=False)['tempBoardSLAVE'].agg('mean')
		target_ak0 = df.groupby(['cpuType', 'cpuBoard', 'remainingTime'], as_index=False)['tempBoardAK0'].agg('mean')
		features.to_pickle(os.path.join(AUTO_ML_DATA_PATH + '/pickle/features_' + filename))
		target_slave.to_pickle(os.path.join(AUTO_ML_DATA_PATH + '/pickle/target_slave_' + filename))
		target_ak0.to_pickle(os.path.join(AUTO_ML_DATA_PATH + '/pickle/target_ak0_' + filename))