def get_service(hass: HomeAssistant, config: ConfigType, discovery_info: DiscoveryInfoType = None) -> PushsaferNotificationService:

class PushsaferNotificationService(BaseNotificationService):
    def __init__(self, private_key: str, is_allowed_path: Callable[[str], bool]):
    
    def send_message(self, message: str = '', **kwargs: Any) -> None:
    
    @classmethod
    def get_base64(cls, filebyte: bytes, mimetype: str) -> Optional[str]:
    
    def load_from_url(self, url: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, auth: Optional[str] = None) -> Optional[str]:
    
    def load_from_file(self, local_path: Optional[str] = None) -> Optional[str]:
