from absl import app
from absl import flags
from constants import _SUPPORTED_APPS, _SUPPORTED_LOGS
from log_processors import MarpleLogProcessor, CadvisorLogProcessor
from log_processors import OpentracingLogProcessor

import json
import os
import sys

FLAGS = flags.FLAGS

flags.DEFINE_enum('app', 'reddit', _SUPPORTED_APPS, 'Application name to '
															'generate dataset.')
flags.DEFINE_enum('log', 'system-wide', _SUPPORTED_LOGS, 'Which logs to use to '
															'generate dataset.')
flags.DEFINE_string('dataset_dir', None, 'Path to faults dataset to featurize.')														

def main(unused_argv):
	if FLAGS.dataset_dir:
		directories = [name for name in os.listdir(FLAGS.dataset_dir) 
						if os.path.isdir(os.path.join(FLAGS.dataset_dir, name))]
		absolute_paths = [os.path.join(FLAGS.dataset_dir, name) for name in \
																	directories]
		if FLAGS.log == 'opentracing' or FLAGS.log == 'system-wide':
			opentracing_processor = OpentracingLogProcessor(FLAGS.app)
			opentracing_processor.generate_opentracing_keys(absolute_paths)
		
		children = []
		for absolute_path in absolute_paths:
			child = os.fork()
			if child==0:
				metadata = None
				with open(os.path.join(absolute_path, 'metadata.json'),'r') as \
																	input_file:
					metadata = json.load(input_file)
				fault = metadata['fault_id']
				if FLAGS.log == 'marple' or FLAGS.log == 'system-wide':
					marple_processor = MarpleLogProcessor(FLAGS.app)
					parsed_log_file = os.path.join(absolute_path, 
							'marple_parsed_{}.joblib'.format(fault))
					try:
						marple_processor.featurize_log_series(absolute_path, 
										absolute_path, parsed_log_file, fault)
					except:
						print('absolute_path: marple')

				if FLAGS.log == 'opentracing' or FLAGS.log == 'system-wide':
					opentracing_processor = OpentracingLogProcessor(FLAGS.app)
					parsed_log_file = os.path.join(absolute_path, 
										'opentracing_extracted_features.json')
					opentracing_processor.featurize_log_series(absolute_path, 
											absolute_path, parsed_log_file, fault)

				if FLAGS.log == 'cadvisor' or FLAGS.log == 'system-wide':
					cadvisor_processor = CadvisorLogProcessor(FLAGS.app)
					parsed_data_path = metadata["parsed_data_path"]
					parsed_log_file = os.path.join(parsed_data_path, 
							'cadvisor_parsed_{}.json'.format(fault))
					cadvisor_processor.featurize_log_series(absolute_path, 
										absolute_path, parsed_log_file, fault)
				sys.exit(0)
			else:
				continue
		for child in children:
			os.waitpid(child, 0)

# run the application
if __name__ == "__main__":
	app.run(main)