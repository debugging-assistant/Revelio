from feature_vectors_pb2 import FeatureVector, RankedMarpleFeatureVector
from feature_vectors_pb2 import RankedCPUFeatureVector
from feature_vectors_pb2 import RankedMemoryFeatureVector
from feature_vectors_pb2 import StatisticalMeasures
from typing import List

def parse_ip(ip: str) -> str:
	ip_num = int(ip)
	return '%d.%d.%d.%d' % (ip_num >> 24, (ip_num >> 16) & 0xff, 
											(ip_num >> 8) & 0xff, ip_num & 0xff)

def app_ip_at_logging_port(port: int, app: str) -> str:
	offset = int(port) - 10090
	if offset < 0:
		offset = int(port) - 9090
	if app=="reddit" and offset==5:
		return "10.0.20.2"
	return "10.0.{}.2".format(offset)

def get_statistical_measures(stat: StatisticalMeasures) -> List[float]:
	values = []
	for descriptor in stat.DESCRIPTOR.fields:
		values.append(getattr(stat, descriptor.name))
	return values

def append_ranked_marple_features(vec: FeatureVector,
						ranked_vec: RankedMarpleFeatureVector,
						with_rank: bool = True) -> FeatureVector:
	marple_features = ranked_vec.vec
	for descriptor in marple_features.DESCRIPTOR.fields:
		if descriptor.name == "queue_stats":
			vec.values.extend(get_statistical_measures(getattr(marple_features, 
															descriptor.name)))
			continue
		vec.values.append(getattr(marple_features, descriptor.name))
	if with_rank:
		vec.values.append(ranked_vec.rank_queue_depth)
		vec.values.append(ranked_vec.rank_tx_rx_ratio)

def append_ranked_cpu_features(vec: FeatureVector,
						ranked_vec: RankedCPUFeatureVector,
						with_rank: bool = True) -> FeatureVector:
	cpu_features = ranked_vec.vec
	for descriptor in cpu_features.DESCRIPTOR.fields:
		if descriptor.name == "cpu_stats":
			vec.values.extend(get_statistical_measures(getattr(cpu_features, 
															descriptor.name)))
			continue
		vec.values.append(getattr(cpu_features, descriptor.name))
	if with_rank:
		vec.values.append(ranked_vec.rank_avg_utilization)

def append_ranked_mem_features(vec: FeatureVector,
						ranked_vec: RankedMemoryFeatureVector,
						with_rank: bool = True) -> FeatureVector:
	mem_features = ranked_vec.vec
	for descriptor in mem_features.DESCRIPTOR.fields:
		if descriptor.name == "mem_stats":
			vec.values.extend(get_statistical_measures(getattr(mem_features, 
															descriptor.name)))
			continue
		vec.values.append(getattr(mem_features, descriptor.name))
	if with_rank:
		vec.values.append(ranked_vec.rank_avg_utilization)