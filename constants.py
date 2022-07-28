# Window around first and last received user reports in an experiment to parse 
# timestamped logs.
_TIME_WINDOW_SECONDS = 60

# Applications for which log processing is implemented.
_SUPPORTED_APPS = ["reddit", "sockshop"]

# Logging tools for which log processing is implemented.
_SUPPORTED_LOGS = ["cadvisor", "marple", "opentracing", "system-wide"]

# Port numbers marple uses to collect logs from P4 switches in the testbed.
_MARPLE_COLLECTION_PORTS = {
	"reddit": list(range(9090,9096)) + list(range(10090,10096)),
	"sockshop": list(range(9090,9105)) + list(range(10090,10105))
}

# Host names cAdvisor uses to collect logs from containers in the testbed.
_HOSTNAMES = {
	"reddit": ["mn.h1","mn.h2","mn.h8","mn.h9","mn.h10","mn.h27"],
	"sockshop": ["mn.h{}".format(num) for num in list(range(1,16))],
}

_VALIDATION_SET_EXCLUSIVE_FAULTS = {
	"reddit": ["3","8","37","49"],
	"sockshop": ["2","8","76","65","49"]
}

_TEST_SET_EXCLUSIVE_FAULTS = {
	"reddit": ["1","11","15","26","25","32","38","50","56","63"],
	"sockshop": ["5","7","18","20","25","35","36","47","63","81"]
}

_OPENTRACING_KEYS_FILE = "opentracing_featurization_keys_{}.json"

_MARPLE_PARAMETERIZED_FEATURE_FILE_BUILDER = "marple_parameterized_features_{}_{}.joblib"
_OPENTRACING_FEATURE_FILE_BUILDER = "opentracing_features_{}_{}.pb"
_CPU_PARAMETERIZED_FEATURE_FILE_BUILDER = "parameterized_cpu_features_{}_{}.joblib"
_MEM_PARAMETERIZED_FEATURE_FILE_BUILDER = "parameterized_mem_features_{}_{}.joblib"
_MARPLE_FEATURE_FILE_BUILDER = "marple_features_{}_{}.pb"
_CADVISOR_FEATURE_FILE_BUILDER = "cadvisor_features_{}_{}.pb"

_TRAIN_SET_FILE_BUILDER = "train_set_{}_{}.joblib"
_VALIDATION_SET_FILE_BUILDER = "validation_set_{}_{}.joblib"
_TEST_SEEN_SET_FILE_BUILDER = "test_seen_{}_{}.joblib"
_TEST_EXCLUSIVE_SET_FILE_BUILDER = "test_exclusive_{}_{}.joblib"