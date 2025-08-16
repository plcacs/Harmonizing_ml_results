    def construct_endpoint(self, service_name: str, region_name: str = None) -> dict:
    def get_available_partitions(self) -> List[str]:
    def get_available_endpoints(self, service_name: str, partition_name: str = 'aws', allow_non_regional: bool = False) -> List[str]:
    def __init__(self, endpoint_data: dict):
    def get_available_partitions(self) -> List[str]:
    def get_available_endpoints(self, service_name: str, partition_name: str = 'aws', allow_non_regional: bool = False) -> List[str]:
    def construct_endpoint(self, service_name: str, region_name: str = None, partition_name: str = None) -> Optional[dict]:
    def _endpoint_for_partition(self, partition: dict, service_name: str, region_name: str, force_partition: bool = False) -> Optional[dict]:
    def _region_match(self, partition: dict, region_name: str) -> bool:
    def _resolve(self, partition: dict, service_name: str, service_data: dict, endpoint_name: str) -> dict:
    def _merge_keys(self, from_data: dict, result: dict):
    def _expand_template(self, partition: dict, template: str, service_name: str, endpoint_name: str) -> str:
