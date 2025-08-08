def setup(hass: HomeAssistant, config: ConfigType) -> bool:
def get_minio_endpoint(host: str, port: int) -> str:
class QueueListener(threading.Thread):
    def __init__(self, hass: HomeAssistant):
    def run(self) -> None:
    @property
    def queue(self) -> Queue:
    def stop(self) -> None:
    def start_handler(self, _) -> None:
    def stop_handler(self, _) -> None:
class MinioListener:
    def __init__(self, queue: Queue, endpoint: str, access_key: str, secret_key: str, secure: bool, bucket_name: str, prefix: str, suffix: str, events: str) -> None:
    def start_handler(self, _) -> None:
    def stop_handler(self, _) -> None:
