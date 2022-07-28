class MarpleLog:

	def __init__(self, queue_depth: int, src_ip: str, dst_ip: str,
				src_port: int, dst_port: int, protocol: int, timestamp: int):
		self._queue_depth = queue_depth
		self._src_ip = src_ip
		self._dst_ip = dst_ip
		self._src_port = src_port
		self._dst_port = dst_port
		self._protocol = protocol
		self._timestamp = timestamp