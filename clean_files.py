import os
import json
from log_processors import CadvisorLogProcessor
import time
import sys

dataset = '../per_user_dataset'

folders = [os.path.join(dataset, name) for name in os.listdir(dataset)]
children = []
for folder in folders:
	if len(children) < 8:
		child = os.fork()
		if child > 0:
			children.append(child)
			continue
	else:
		for child in children:
			os.waitpid(child, 0)
		children = []
		child = os.fork()
		if child > 0:
			children.append(child)
			continue
	for name in os.listdir(folder):
		if "opentracing_parsed" in name or "cadvisor_parsed" in name:
			log = None
			with open(os.path.join(folder, name), 'r') as input_file:
				log = json.load(input_file)

			processor = CadvisorLogProcessor("reddit")
			start_timestamp, end_timestamp = processor.get_reports_time_range(folder)
			if "cadvisor_parsed" in name:
				output_log = {}
				for container in log:
					if container in ["cadvisor", "jaeger"]:
						continue
						output_log[container] = log[container].copy()
						output_log[container]["stats"] = []
					for stat in log[container]["stats"]:
						stamp = time.mktime(time.strptime(
														stat["timestamp"][:-4],
														"%Y-%m-%dT%H:%M:%S.%f"))
						if stamp >= start_timestamp and stamp <= end_timestamp:
							output_log[container]["stats"].append(stat)
						elif stamp > end_timestamp:
							break
				with open(os.path.join(folder, name), 'w') as outputfile:
					json.dump(output_log, outputfile)
			else:
				output_log = []
				for trace in log:
					for span in trace["spans"]:
						span_timestamp = span["startTime"]/1000000
						if span_timestamp <= end_timestamp and \
									span_timestamp > start_timestamp:
							output_log.append(trace)
							break
				with open(os.path.join(folder, name), 'w') as outputfile:
					json.dump(output_log, outputfile)
	sys.exit(0)

for child in children:
	os.waitpid(child, 0)