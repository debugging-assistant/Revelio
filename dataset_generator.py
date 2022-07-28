import os
import sys
import json
import numpy as np
from sklearn.preprocessing import normalize

from absl import app
from absl import flags
from constants import _SUPPORTED_APPS, _HOSTNAMES, _MARPLE_COLLECTION_PORTS
from constants import _VALIDATION_SET_EXCLUSIVE_FAULTS, _SUPPORTED_LOGS
from constants import _TEST_SET_EXCLUSIVE_FAULTS
from scipy import spatial
from typing import List 

import constants
import feature_vectors_pb2
import joblib
import utils

FLAGS = flags.FLAGS

flags.DEFINE_enum('app', 'reddit', _SUPPORTED_APPS, 'Application name to '
															'generate dataset.')
flags.DEFINE_enum('log', 'system-wide', _SUPPORTED_LOGS, 'Which logs to use to '
															'generate dataset.')
flags.DEFINE_string('dataset_dir', None, 'Path to faults dataset to featurize.')
flags.DEFINE_string('output_dir', None, 'Directory to save datasets.')		

def gen_system_wide_dataset(dataset_dir: str, app: str) -> None:

	dataset = []
	folders = [os.path.join(dataset_dir, name) for name in os.listdir(dataset_dir)
							if os.path.isdir(os.path.join(dataset_dir, name))]
	for folder in folders:
		with open(os.path.join(folder, 'metadata.json'), 'r') as input_file:
			metadata = json.load(input_file)
		fault = metadata["fault_id"]
		
		datapoint = {}
		
		opentracing_features = feature_vectors_pb2.FeatureVector()
		file_name = constants._OPENTRACING_FEATURE_FILE_BUILDER.format(app, fault)
		with open(os.path.join(folder,file_name),'rb') as input_file:
			opentracing_features.ParseFromString(input_file.read())

		marple_features = feature_vectors_pb2.FeatureVector()
		file_name = constants._MARPLE_FEATURE_FILE_BUILDER.format(app, fault)
		with open(os.path.join(folder,file_name),'rb') as input_file:
			marple_features.ParseFromString(input_file.read())
		
		cadvisor_features = feature_vectors_pb2.FeatureVector()
		file_name = constants._CADVISOR_FEATURE_FILE_BUILDER.format(app, fault)
		with open(os.path.join(folder,file_name),'rb') as input_file:
			cadvisor_features.ParseFromString(input_file.read())


		features = feature_vectors_pb2.FeatureVector()
		features.values.extend(marple_features.values)
		features.values.extend(opentracing_features.values)
		features.values.extend(cadvisor_features.values)
		if len(features.values) > 376:
			print(folder)
			print(len(features.values))
		
		datapoint["features"] = [value for value in features.values]
		datapoint["fault"] = fault
		datapoint["label"] = label
		dataset.append(datapoint)


	rng_state = np.random.get_state()
	np.random.shuffle(dataset)

	train_features = []
	train_labels = []

	validation_features = []
	validation_labels = []

	test_exclusive_features = []
	test_exclusive_labels = []

	test_seen_features = []
	test_seen_labels = []

	validation_overlap = {}
	test_overlap = {}
	train_overlap = {}

	for datapoint in dataset:
		feature = datapoint["features"]
		fault = datapoint["fault"]
		label = datapoint["label"]
		if fault in _VALIDATION_SET_EXCLUSIVE_FAULTS[app]:
			validation_labels.append(label)
			validation_features.append(feature)
		elif fault in _TEST_SET_EXCLUSIVE_FAULTS[app]:
			test_exclusive_labels.append(label)
			test_exclusive_features.append(feature)
		else:
			if fault not in validation_overlap and fault not in test_overlap and len(validation_overlap) < 4:
				validation_overlap[fault] = 1
				train_overlap[fault] = 0
				validation_labels.append(label)
				validation_features.append(feature)
			elif fault not in validation_overlap and fault not in test_overlap and len(test_overlap) < 11:
				test_overlap[fault] = 1
				train_overlap[fault] = 0
				test_seen_labels.append(label)
				test_seen_features.append(feature)
			elif fault not in validation_overlap and fault not in test_overlap:
				train_labels.append(label)
				train_features.append(feature)
			elif fault in validation_overlap:
				if validation_overlap[fault] >= train_overlap[fault]:
					train_overlap[fault] = train_overlap[fault] + 1
					train_labels.append(label)
					train_features.append(feature)
				else:
					validation_overlap[fault] = validation_overlap[fault] + 1
					validation_labels.append(label)
					validation_features.append(feature)
			elif fault in test_overlap:
				if test_overlap[fault] >= train_overlap[fault]:
					train_overlap[fault] = train_overlap[fault] + 1
					train_labels.append(label)
					train_features.append(feature)
				else:
					test_overlap[fault] = test_overlap[fault] + 1
					test_seen_labels.append(label)
					test_seen_features.append(feature)

	train_set = {}
	train_set["features"] = train_features
	train_set["labels"] = train_labels
	
	validation_set = {}
	validation_set["features"] = validation_features
	validation_set["labels"] = validation_labels
	
	test_seen_set = {}
	test_seen_set["features"] = test_seen_features
	test_seen_set["labels"] = test_seen_labels
	
	test_exclusive_set = {}
	test_exclusive_set["features"] = test_exclusive_features
	test_exclusive_set["labels"] = test_exclusive_labels

	print("Training samples: {}".format(len(train_set["labels"])))
	print("Validation samples: {}".format(len(validation_set["labels"])))
	print("Exclusive test samples: {}".format(len(test_exclusive_set["labels"])))
	print("Repeat test samples: {}".format(len(test_seen_set["labels"])))

	file_name = constants._TRAIN_SET_FILE_BUILDER.format(FLAGS.log, app)
	with open(os.path.join(FLAGS.output_dir, file_name), 'wb') as output_file:
		joblib.dump(train_set, output_file)
		
	file_name = constants._VALIDATION_SET_FILE_BUILDER.format(FLAGS.log, app)
	with open(os.path.join(FLAGS.output_dir, file_name), 'wb') as output_file:
		joblib.dump(validation_set, output_file)
		
	file_name = constants._TEST_SEEN_SET_FILE_BUILDER.format(FLAGS.log, app)
	with open(os.path.join(FLAGS.output_dir, file_name), 'wb') as output_file:
		joblib.dump(test_seen_set, output_file)
		
	file_name = constants._TEST_EXCLUSIVE_SET_FILE_BUILDER.format(FLAGS.log, app)
	with open(os.path.join(FLAGS.output_dir, file_name), 'wb') as output_file:
		joblib.dump(test_exclusive_set, output_file)

def main(unused_argv):
	if FLAGS.log == 'system-wide' and FLAGS.dataset_dir:
		gen_system_wide_dataset(FLAGS.dataset_dir, FLAGS.app)

# run the application
if __name__ == "__main__":
	app.run(main)