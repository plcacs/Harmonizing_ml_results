def to_python(value: str) -> AddressBytes:
def to_url(value: AddressBytes) -> AddressHex:
def _serialize(value: Any, attr: str, obj: Any, **kwargs: Any) -> str:
def _deserialize(self, value: Any, attr: str, data: Dict[str, Any], **kwargs: Any) -> AddressBytes:
def _serialize(value: Any, attr: str, obj: Any, **kwargs: Any) -> str:
def _deserialize(self, value: Any, attr: str, data: Dict[str, Any], **kwargs: Any) -> bytes:
def _serialize(value: Any, attr: str, obj: Any, **kwargs: Any) -> str:
def _deserialize(self, value: Any, attr: str, data: Dict[str, Any], **kwargs: Any) -> bytes:
def _serialize(value: Any, attr: str, obj: Any, **kwargs: Any) -> str:
def _deserialize(self, value: Any, attr: str, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
def __init__(self, meta: Any, ordered: bool) -> None:
def make_object(self, data: Dict[str, Any], **kwargs: Any) -> Any:
def wrap_data_envelope(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
def unwrap_data_envelope(self, data: Dict[str, Any], **kwargs: Any) -> Any:
def make_object(self, data: Dict[str, Any], **kwargs: Any) -> Any:
def serialize(self, chain_state: ChainState, event: TimestampedEvent) -> Dict[str, Any]:
