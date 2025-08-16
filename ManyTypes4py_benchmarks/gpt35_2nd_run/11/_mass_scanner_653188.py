from queue import Queue
from typing import TypeAlias

ServerScanRequestsQueueType: TypeAlias = 'Queue[Union[NoMoreServerScanRequestsSentinel, Tuple[ServerScanRequest, ServerTlsProbingResult]]]'
ServerScanResultsQueueType: TypeAlias = 'Queue[Union[NoMoreServerScanRequestsSentinel, ServerScanResult]]'
