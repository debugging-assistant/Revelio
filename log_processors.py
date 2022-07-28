from abc import ABC, abstractmethod
from collections import defaultdict
from constants import _TIME_WINDOW_SECONDS, _MARPLE_COLLECTION_PORTS, _HOSTNAMES
from constants import _OPENTRACING_KEYS_FILE
from data_structures import MarpleLog
from datetime import datetime
from typing import List, TextIO, Tuple
from utils import parse_ip, app_ip_at_logging_port

import errno
import joblib
import json
import math
import numpy as np
import operator
import os
import tarfile
import time
import utils

import feature_vectors_pb2
from feature_vectors_pb2 import StatisticalMeasures


class LogProcessor(ABC):

	def __init__(self, app):
		self._app = app

	@classmethod
	def decompress_logs(cls, archive_file: str, output_dir: str) -> None:
		"""Extracts compressed archives of log files.
		Only accepts archives achieved by compressing with GNU zip utility into 
		a tar file (aka archives with .tar.gz extension).

		Args:
			archive_file: Path to the archive to the decompressed.
			output_dir: Directory to extract log files into.

		Raises:
			FileNotFoundError: Archive file or Output directory doesn't exist.
			TypeError: Invalid type of archive file input.
		"""
		if not os.path.isdir(output_path):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 output_path)
		if os.path.isfile(archive_path):
			if archive_path.endswith('.tar.gz'):
				tar = tarfile.open(archive_path, 'r:gz')
				tar.extractall(path=output_path)
			else:
				raise TypeError("Invalid file extension for archive, only"
				" .tar.gz allowed.")
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 archive_file)

	@abstractmethod
	def parse_log_series(self, folder: str, fault: str) -> None:
		"""API method to implement log-specific parsing.
		Outputs series of logs that can be used for featurization.

		Args:
			folder: Path to directory with raw logs from an experiment.

		Returns:
			parsed_log_file: Path to the file containing the parsed data from \
							 raw logs.
			fault_id: ID of the fault injected during the experiment.
		"""
		pass

	@abstractmethod
	def featurize_log_series(self, reports_folder: str, output_folder: str, 
									parsed_log_file: str, fault: str) -> None:
		"""API method to implement log-specific featurization from parsed logs.
		Outputs the generated features for each experiment.
		"""
		pass

	def get_reports_time_range(self, folder: str) -> Tuple[datetime, datetime]:
		"""Reads user-reports in the given folder and outputs the start and end
		timestamps of users interaction with the application.

		Args:
			folder: Path to the folder that contains an experiment's data.

		Returns:
			start_timestamp: Indicates the receipt of first user bug report.
			end_timestamp: Indicates the receipt of last user bug report.

		Raises:
			FileNotFoundError: The provided path is not a directory.
		"""
		if os.path.isdir(folder):
			base_name = os.path.basename(os.path.normpath(folder))
			start_timestamp = -1
			end_timestamp = -1
			reports = [name for name in os.listdir(folder) if name.startswith(
								'report_') and name.endswith('.json')]
			for report_file in reports:
				with open(os.path.join(folder,report_file),'r') as input_file:
					issue_report = json.load(input_file)
					first_answer_timestamp = issue_report[0]["timestamp"]
					last_answer_timestamp = issue_report[-1]["timestamp"]
					if start_timestamp == -1 and end_timestamp == -1:
						start_timestamp = first_answer_timestamp
						end_timestamp = last_answer_timestamp
					if first_answer_timestamp < start_timestamp:
						start_timestamp = first_answer_timestamp
					if end_timestamp < last_answer_timestamp:
						end_timestamp = last_answer_timestamp
			if start_timestamp == -1 and end_timestamp == -1:
				start_timestamp = _TIME_WINDOW_SECONDS
				end_timestamp = int(time.time())
			start_timestamp -= _TIME_WINDOW_SECONDS
			end_timestamp += _TIME_WINDOW_SECONDS
			start_timestamp = datetime.utcfromtimestamp(start_timestamp)
			end_timestamp = datetime.utcfromtimestamp(end_timestamp)
			return start_timestamp, end_timestamp
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 folder)

	def compute_standard_stats(self, series: List[float]
													) -> StatisticalMeasures:
		output = StatisticalMeasures()
		output.avg = np.mean(series)
		output.max = max(series)
		output.median = np.percentile(series, 50)
		output.firstq = np.percentile(series, 25)
		output.thirdq = np.percentile(series, 75)
		output.std_dev = np.std(series)
		return output

class MarpleLogProcessor(LogProcessor):

	def parse_log_series(self, folder: str, fault: str) -> None:
		"""
		fill this up.
		"""
		if os.path.isdir(folder):
			per_switch_logs = defaultdict(list)
			base_name = os.path.basename(os.path.normpath(folder))
			# Each switch's logs are stored in a directory within this folder.
			# We find all such directories and identify the switches by the
			# folder names.
			per_switch_log_dirs = [dir_name for dir_name in os.listdir(folder) 
				if os.path.isdir(os.path.join(folder,dir_name))]
			for switch_log_dir in per_switch_log_dirs:
				switch_id = switch_log_dir[7:]
				switch_log_path = os.path.join(folder,switch_log_dir)
				cache_timestamps_path = os.path.join(switch_log_path,
													'cache_timestamps.json')
				log_sequence_path = os.path.join(switch_log_path,
													'p4logs.txt')

				log_indices = [int(log_file[:log_file.index('.txt')]) \
							for log_file in os.listdir(switch_log_path) \
							if log_file!="p4logs.txt" and \
							log_file.endswith('.txt') and \
							os.path.isfile(os.path.join(switch_log_path,
							log_file))]
				log_indices.sort()

				cache_timestamps = list()
				if os.path.isfile(cache_timestamps_path):
					with open(cache_timestamps_path,'r') as input_file:
						try:
							cache_timestamps = json.load(input_file)
						except:
							cache_timestamps = list()

				if os.path.isfile(log_sequence_path):
					input_file = open(log_sequence_path,'r')
					sequence, _ = self.parse_log_lines(input_file, cache_timestamps)
					per_switch_logs[switch_id].append(sequence)
					with open(cache_timestamps_path,'w') as output:
						json.dump(cache_timestamps, output)

				for log_index in log_indices:
					log_file_path = os.path.join(switch_log_path,
												str(log_index)+'.txt')
					logfile = open(log_file_path,'r')
					sequence, lines = self.parse_log_lines(logfile, cache_timestamps)
					per_switch_logs[switch_id].append(sequence)
					
					if os.path.isfile(log_sequence_path):
						with open(log_sequence_path,'a') as output:
							output.writelines(lines)
					else:
						with open(log_sequence_path,'w') as output:
							output.writelines(lines)
					with open(cache_timestamps_path,'w') as output:
						json.dump(cache_timestamps, output)
					os.remove(log_file_path)
			output_file = 'marple_parsed_{}.joblib'.format(fault)
			with open(os.path.join(folder, output_file),'wb') as output:
				joblib.dump(per_switch_logs, output)
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 folder)

	def featurize_log_series(self, reports_folder: str, output_folder: str, 
									parsed_log_file: str, fault: str) -> None:
		"""
		fill this up.
		"""
		if os.path.isfile(parsed_log_file):
			with open(parsed_log_file,'rb') as input_file:
				marple_log = joblib.load(input_file)
				per_logging_port_features = {}
				avg_queue_depths = {}
				tx_rx_ratios = {}
				for logging_port_raw in marple_log:
					logging_port = int(logging_port_raw)
					queues = []
					l3_pkt_counts = 0
					l2_pkt_counts = 0
					to_packets = 0
					from_packets = 0
					app_ip = app_ip_at_logging_port(logging_port, self._app)
					for sequence in marple_log[logging_port_raw]:
						for data_point in sequence:
							queues.append(data_point._queue_depth)
							if data_point._src_ip == 0 and data_point._dst_ip == 0:
								l2_pkt_counts += 1
							else:
								l3_pkt_counts += 1
								if parse_ip(data_point._src_ip) == app_ip:
									from_packets += 1
								elif parse_ip(data_point._dst_ip) == app_ip:
									to_packets += 1
					if not queues:
						continue
					stats = self.compute_standard_stats(queues)
					tx_rx_ratio = 0.0
					if to_packets != 0:
						tx_rx_ratio = float(from_packets)/to_packets
					
					vec = feature_vectors_pb2.MarpleFeatureVector()
					vec.queue_stats.CopyFrom(stats)
					vec.l3_pkt_counts = l3_pkt_counts
					vec.from_packets = from_packets
					vec.to_packets = to_packets
					vec.tx_rx_ratio = tx_rx_ratio
					per_logging_port_features[logging_port] = vec

					avg_queue_depths[logging_port] = stats.avg
					tx_rx_ratios[logging_port] = tx_rx_ratio
				sorted_avg_queue_depths = sorted(avg_queue_depths.items(), 
									key=operator.itemgetter(1), reverse=True)
				ranked_ports_by_avg_queue_depth = [port for port, _ in 
														sorted_avg_queue_depths]
				sorted_tx_rx_ratios = sorted(tx_rx_ratios.items(), 
													key=operator.itemgetter(1))
				ranked_ports_by_tx_rx_ratios = [port for port, _ in 
															sorted_tx_rx_ratios]
				network_feature_vec = feature_vectors_pb2.FeatureVector()
				per_switch_ranked_features = {}
				rank=0
				for port in ranked_ports_by_avg_queue_depth:
					rank+=1
					ranked_vec = feature_vectors_pb2.RankedMarpleFeatureVector()
					ranked_vec.vec.CopyFrom(per_logging_port_features[port])
					ranked_vec.rank_queue_depth = rank
					ranked_vec.rank_tx_rx_ratio = \
									ranked_ports_by_tx_rx_ratios.index(port)+1
					per_switch_ranked_features[port] = ranked_vec
					utils.append_ranked_marple_features(
												network_feature_vec, ranked_vec)

				for port in _MARPLE_COLLECTION_PORTS[self._app]:
					if port in per_switch_ranked_features:
						continue
					ranked_vec = feature_vectors_pb2.RankedMarpleFeatureVector()
					ranked_vec.rank_queue_depth = rank + \
										len(_MARPLE_COLLECTION_PORTS[self._app])
					ranked_vec.rank_tx_rx_ratio = rank + \
										len(_MARPLE_COLLECTION_PORTS[self._app])
					per_switch_ranked_features[port] = ranked_vec
					utils.append_ranked_marple_features(
												network_feature_vec, ranked_vec)

				file_name = 'marple_features_{}_{}.pb'.format(self._app, fault)
				with open(os.path.join(output_folder,file_name),'wb') as output_file:
					output_file.write(network_feature_vec.SerializeToString())

				file_name = 'marple_parameterized_features_{}_{}.joblib'.format(
																self._app, fault)
				with open(os.path.join(output_folder,file_name),'wb') as output_file:
					joblib.dump(per_switch_ranked_features, output_file)
		else:
			network_feature_vec = feature_vectors_pb2.FeatureVector()
			per_switch_ranked_features = {}
			rank = 0
			for port in _MARPLE_COLLECTION_PORTS[self._app]:
				if port in per_switch_ranked_features:
					continue
				ranked_vec = feature_vectors_pb2.RankedMarpleFeatureVector()
				ranked_vec.rank_queue_depth = rank + \
									len(_MARPLE_COLLECTION_PORTS[self._app])
				ranked_vec.rank_tx_rx_ratio = rank + \
									len(_MARPLE_COLLECTION_PORTS[self._app])
				per_switch_ranked_features[port] = ranked_vec
				utils.append_ranked_marple_features(
											network_feature_vec, ranked_vec)

			file_name = 'marple_features_{}_{}.pb'.format(self._app, fault)
			with open(os.path.join(output_folder,file_name),'wb') as output_file:
				output_file.write(network_feature_vec.SerializeToString())

			file_name = 'marple_parameterized_features_{}_{}.joblib'.format(
															self._app, fault)
			with open(os.path.join(output_folder,file_name),'wb') as output_file:
				joblib.dump(per_switch_ranked_features, output_file)
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 parsed_log_file)

	def parse_log_lines(self, input_file: TextIO, timestamps: List[int]
										) -> Tuple[List[MarpleLog], List[str]]:
		"""
		fill this up.
		"""
		output_log_sequence = list()
		log_sequence = list()
		writeable_sequence = list()
		try:
			for line in input_file:
				if line.startswith('reg'):
					log_sequence.append(line[line.index('=')+2:-1])
				if line.startswith('times'):
					timestamp = line[line.index('=')+2:-1]
					log_sequence.append(timestamp)
					timestamp = int(timestamp)
					log_obj = MarpleLog(int(log_sequence[0]),
										log_sequence[1],
										log_sequence[2],
										int(log_sequence[3]),
										int(log_sequence[4]),
										int(log_sequence[5]),
										timestamp)
					output_log_sequence.append(log_obj)
					if timestamp != 0 and timestamp not in timestamps:
						timestamps.append(timestamp)
					writeable_sequence.extend(log_sequence)
					log_sequence = list()
		except:
			log_sequence = list()
		return output_log_sequence, writeable_sequence

class OpentracingLogProcessor(LogProcessor):

	def parse_log_series(self, folder: str, fault: str) -> None:
		"""
		fill this up.
		"""
		if os.path.isdir(folder):
			base_name = os.path.basename(os.path.normpath(folder))
			trace_start_timestamp, trace_end_timestamp = \
											self.get_reports_time_range(folder)

			opentracing_log = list()
			traces_to_filter = list()
			file_nos = {}
			log_files = [log_file for log_file in os.listdir(folder) \
					if log_file == 'traces.json']
			for log_file in log_files:
				traces = None
				with open(os.path.join(folder,log_file),'r') as input_file:
					traces = json.load(input_file)
				for trace in traces["data"]:
					if trace["traceID"] in traces_to_filter:
						continue
					traces_to_filter.append(trace["traceID"])
					for span in trace["spans"]:
						span_timestamp = span["startTime"]/1000000
						span_timestamp = datetime.utcfromtimestamp(
															span_timestamp)
						if span_timestamp <= trace_end_timestamp and \
									span_timestamp > trace_start_timestamp:
							opentracing_log.append(span)
			output_file = 'opentracing_parsed_{}.json'.format(fault)
			with open(os.path.join(folder, output_file),'w') as output:
				json.dump(opentracing_log, output)
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 folder)

	def featurize_log_series(self, reports_folder: str, output_folder: str,
									parsed_log_file: str, fault: str) -> None:
		"""API method to implement log-specific featurization from parsed logs.
		Outputs the generated features for each experiment.
		"""
		key_file = _OPENTRACING_KEYS_FILE.format(self._app)
		if not os.path.isfile(key_file):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 key_file)

		tag_ids = None
		function_ids = None
		with open(key_file,'r') as input_file:
			featurization_map = json.load(input_file)
			tag_ids = featurization_map["tag_ids"]
			function_ids = featurization_map["function_ids"]

		tag_counts = None
		function_durations = None
		with open(parsed_log_file,'r') as input_file:
			extracted_features = json.load(input_file)
			tag_counts = extracted_features["tag_counts"]
			function_durations = extracted_features["function_durations"]

		tag_counts_feature_vec = [0]*len(tag_ids)
		durations_feature_vec = [0]*len(function_ids)
		for function, durations in function_durations.items():
			durations_feature_vec[function_ids[function]] = np.mean(durations)
		for tag, count in tag_counts.items():
			tag_counts_feature_vec[tag_ids[tag]] = count

		feature_vec = feature_vectors_pb2.FeatureVector()
		feature_vec.values.extend(tag_counts_feature_vec)
		feature_vec.values.extend(durations_feature_vec)

		file_name = 'opentracing_features_{}_{}.pb'.format(self._app, fault)
		with open(os.path.join(output_folder,file_name),'wb') as output_file:
			output_file.write(feature_vec.SerializeToString())

	def generate_opentracing_keys(self, folders: List[str]) -> None:
		tag_ids = defaultdict(lambda: -1)
		function_ids = defaultdict(lambda: -1)

		for folder in folders:
			tag_counts = defaultdict(int)
			function_tags = defaultdict(list)
			function_durations = defaultdict(list)
			if os.path.isdir(folder):
				logs = [name for name in os.listdir(folder) 
									if name.startswith('opentracing_parsed_')]
				if len(logs) == 1:
					with open(os.path.join(folder, logs[0]),'r') as input_file:
						opentracing_log = json.load(input_file)
				else:
					raise ValueError("There can only be one opentracing log "
															"file per folder")
				trace_start_timestamp, trace_end_timestamp = \
											self.get_reports_time_range(folder)	
				for span in opentracing_log:
					span_timestamp = span["startTime"]/1000000
					span_timestamp = datetime.utcfromtimestamp(
															span_timestamp)
					if span_timestamp < trace_start_timestamp:
						continue
					if span_timestamp > trace_end_timestamp:
						break
					function_name = span["operationName"]
					if function_name not in function_ids:
						function_ids[function_name] = len(function_ids)	
					function_durations[function_name].append(span["duration"])
					for tag in span["tags"]:
						if tag["key"] not in function_tags[function_name]:
							function_tags[function_name].append(tag["key"])
						tag_name = function_name + tag["key"]
						if tag_name not in tag_ids:
							tag_ids[tag_name] = len(tag_ids)
						tag_counts[tag_name] += 1
				file_name = "opentracing_extracted_features.json"
				with open(os.path.join(folder, file_name),'w') as output_file:
					mapping = {}
					mapping["tag_counts"] = tag_counts
					mapping["function_durations"] = function_durations
					mapping["function_tags"] = function_tags
					json.dump(mapping, output_file)
			else:
				raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
				 folder)
		
		file_name = _OPENTRACING_KEYS_FILE.format(self._app) 
		with open(file_name,'w') as output_file:
			mapping = {}
			mapping["function_ids"] = function_ids
			mapping["tag_ids"] = tag_ids
			json.dump(mapping, output_file)

class CadvisorLogProcessor(LogProcessor):

	def parse_log_series(self, folder: str, fault: str) -> None:
		if os.path.isdir(folder):
			base_name = os.path.basename(os.path.normpath(folder))
			trace_start_timestamp, trace_end_timestamp = \
											self.get_reports_time_range(folder)
			cadvisor_logs = [name for name in os.listdir(folder) \
					if name.startswith('cadvisor_') and name.endswith('.json') \
													and "parsed" not in name]
			cadvisor_log_index = [int(name[name.index("_")+1:name.index(".")]) \
													for name in cadvisor_logs]
			sorted_log_indices = sorted(cadvisor_log_index)
			time_series = {}
			timestamps = defaultdict(list)
			for index in sorted_log_indices:
				with open(os.path.join(folder,"cadvisor_"+str(index)+".json"), 
															'r') as input_file:
					try:
						content = json.load(input_file)
					except:
						continue
					for key in content:
						if "cadvisor" not in str(content[key]["aliases"][0]):
							name = str(content[key]["aliases"][0])
							if name not in time_series:
								time_series[name] = content[key].copy()
								time_series[name]["stats"] = []
							else:
								for obj in content[key]["stats"]:
									try:
										stamp = datetime.strptime(
													obj["timestamp"][:-4],
														"%Y-%m-%dT%H:%M:%S.%f")
									except:
										continue
									if stamp not in timestamps[name] and \
										stamp >= trace_start_timestamp and \
										stamp <= trace_end_timestamp:
										time_series[name]["stats"].append(obj)
										timestamps[name].append(stamp)
									elif stamp >= trace_end_timestamp:
										break
			output_file = 'cadvisor_parsed_{}.json'.format(fault)
			with open(os.path.join(folder,output_file),'w') as output:
				json.dump(time_series,output)
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 folder)

	def featurize_log_series(self, reports_folder: str, output_folder: str, 
									parsed_log_file: str, fault: str) -> None:
		"""API method to implement log-specific featurization from parsed logs.
		Outputs the generated features for each experiment.
		"""
		if os.path.isfile(parsed_log_file):
			with open(parsed_log_file,'r') as input_file:
				cadvisor_log = json.load(input_file)
			trace_start_timestamp, trace_end_timestamp = \
									self.get_reports_time_range(reports_folder)

			per_container_cpu_features = {}
			per_container_mem_features = {}
			avg_cpu_utils = {}
			avg_mem_utils = {}
			for container in cadvisor_log:
				time_vs_usage = []
				if container in ["cadvisor", "jaeger"]:
					continue
				for stat in cadvisor_log[container]["stats"]:
					timestamp = datetime.strptime(stat["timestamp"][:-4], 
														"%Y-%m-%dT%H:%M:%S.%f")
					if timestamp < trace_start_timestamp:
						continue
					if timestamp > trace_end_timestamp:
						break
					cpu_usage = stat["cpu"]["usage"]["total"]
					mem_usage = stat["memory"]["usage"]
					mem_usage = (float)(mem_usage)/(1024*1024)
					time_vs_usage.append((timestamp,cpu_usage,mem_usage))
				
				time_vs_usage = sorted(time_vs_usage, key=lambda x: x[0])
				cpu_utilization = list()
				mem_utilization = list()
				for i in range(1,len(time_vs_usage)):
					diff = time_vs_usage[i][0] - time_vs_usage[i-1][0]
					diff_nanos = diff.total_seconds()*1000000000
					if diff_nanos == 0:
						continue
					cpu_diff = time_vs_usage[i][1] - time_vs_usage[i-1][1]
					usage_level = (float)(cpu_diff)/diff_nanos
					if usage_level<0:
						cpu_utilization.append(0)
					else:
						cpu_utilization.append(usage_level)
					mem_utilization.append(time_vs_usage[i][2])
				if not cpu_utilization:
					continue
					
				stats = self.compute_standard_stats(cpu_utilization)
				
				cpu_feature_vec = feature_vectors_pb2.CPUFeatureVector()
				cpu_feature_vec.cpu_stats.CopyFrom(stats)
				per_container_cpu_features[container] = cpu_feature_vec
				avg_cpu_utils[container] = stats.avg
				
				stats = self.compute_standard_stats(mem_utilization)
				
				mem_feature_vec = feature_vectors_pb2.MemoryFeatureVector()
				mem_feature_vec.mem_stats.CopyFrom(stats)
				per_container_mem_features[container] = mem_feature_vec
				avg_mem_utils[container] = stats.avg

			sorted_avg_cpu_utils = sorted(avg_cpu_utils.items(), 
													key=operator.itemgetter(1))
			ranked_containers_by_cpu = [container for container, _ in 
														sorted_avg_cpu_utils]
			sorted_avg_mem_utils = sorted(avg_mem_utils.items(), 
													key=operator.itemgetter(1))
			ranked_containers_by_mem = [container for container, _ in 
														sorted_avg_mem_utils]
			
			cadvisor_feature_vec = feature_vectors_pb2.FeatureVector()
			per_container_ranked_cpu_features = {}
			per_container_ranked_mem_features = {}
			
			rank = 0
			for container in ranked_containers_by_cpu:
				rank += 1
				ranked_vec = feature_vectors_pb2.RankedCPUFeatureVector()
				ranked_vec.vec.CopyFrom(per_container_cpu_features[container])
				ranked_vec.rank_avg_utilization = rank
				per_container_ranked_cpu_features[container] = ranked_vec
				utils.append_ranked_cpu_features(
											cadvisor_feature_vec, ranked_vec)
				
			for container in _HOSTNAMES[self._app]:
				if container in per_container_ranked_cpu_features:
					continue
				ranked_vec = feature_vectors_pb2.RankedCPUFeatureVector()
				ranked_vec.rank_avg_utilization = rank+len(_HOSTNAMES[self._app])
				per_container_ranked_cpu_features[container] = ranked_vec
				utils.append_ranked_cpu_features(
										cadvisor_feature_vec, ranked_vec)
			
			
			rank = 0
			for container in ranked_containers_by_mem:
				rank += 1
				ranked_vec = feature_vectors_pb2.RankedMemoryFeatureVector()
				ranked_vec.vec.CopyFrom(per_container_mem_features[container])
				ranked_vec.rank_avg_utilization = rank
				per_container_ranked_mem_features[container] = ranked_vec
				utils.append_ranked_mem_features(
											cadvisor_feature_vec, ranked_vec)
				
			for container in _HOSTNAMES[self._app]:
				if container in per_container_ranked_mem_features:
					continue
				ranked_vec = feature_vectors_pb2.RankedMemoryFeatureVector()
				ranked_vec.rank_avg_utilization = rank+len(_HOSTNAMES[self._app])
				per_container_ranked_mem_features[container] = ranked_vec
				utils.append_ranked_mem_features(
										cadvisor_feature_vec, ranked_vec)

			file_name = 'cadvisor_features_{}_{}.pb'.format(self._app, fault)
			with open(os.path.join(output_folder, file_name),'wb') as output_file:
				output_file.write(cadvisor_feature_vec.SerializeToString())
			
			file_name = 'parameterized_cpu_features_{}_{}.joblib'.format(
																self._app, fault)
			with open(os.path.join(output_folder, file_name),'wb') as output_file:
				joblib.dump(per_container_ranked_cpu_features, output_file)
			
			file_name = 'parameterized_mem_features_{}_{}.joblib'.format(
																self._app, fault)
			with open(os.path.join(output_folder, file_name),'wb') as output_file:
				joblib.dump(per_container_ranked_mem_features, output_file)
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
			 parsed_log_file)