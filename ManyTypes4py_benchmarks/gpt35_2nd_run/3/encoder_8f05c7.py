from typing import Any, Dict, List, Tuple

class MultipartEncoder:
    def __init__(self, fields: Dict[str, Any], boundary: str = None, encoding: str = 'utf-8') -> None:
    def __repr__(self) -> str:
    @property
    def len(self) -> int:
    def _calculate_length(self) -> int:
    def _calculate_load_amount(self, read_size: int) -> int:
    def _load(self, amount: int) -> None:
    def _next_part(self) -> Any:
    def _iter_fields(self) -> Any:
    def _prepare_parts(self) -> None:
    def _write(self, bytes_to_write: bytes) -> int:
    def _write_boundary(self) -> int:
    def _write_closing_boundary(self) -> int:
    def _write_headers(self, headers: str) -> int:
    @property
    def content_type(self) -> str:
    def to_string(self) -> bytes:
    def read(self, size: int = -1) -> bytes

class MultipartEncoderMonitor:
    def __init__(self, encoder: MultipartEncoder, callback: Any = None) -> None:
    @classmethod
    def from_fields(cls, fields: Dict[str, Any], boundary: str = None, encoding: str = 'utf-8', callback: Any = None) -> MultipartEncoderMonitor:
    @property
    def content_type(self) -> str:
    def to_string(self) -> bytes:
    def read(self, size: int = -1) -> bytes

def encode_with(string: str, encoding: str) -> bytes:
def readable_data(data: Any, encoding: str) -> Any:
def total_len(o: Any) -> int:
def reset(buffer: Any) -> Any:
def coerce_data(data: Any, encoding: str) -> Any:
def to_list(fields: Any) -> List[Tuple[str, Any]]:

class Part:
    def __init__(self, headers: bytes, body: Any) -> None:
    @classmethod
    def from_field(cls, field: Any, encoding: str) -> Part:
    def bytes_left_to_write(self) -> bool:
    def write_to(self, buffer: Any, size: int) -> int

class CustomBytesIO(io.BytesIO):
    def __init__(self, buffer: bytes = None, encoding: str = 'utf-8') -> None:
    @property
    def len(self) -> int:
    def append(self, bytes: bytes) -> int:
    def smart_truncate(self) -> None

class FileWrapper:
    def __init__(self, file_object: Any) -> None:
    @property
    def len(self) -> int:
    def read(self, length: int = -1) -> bytes

class FileFromURLWrapper:
    def __init__(self, file_url: str, session: Any = None) -> None:
    def _request_for_file(self, file_url: str) -> Any:
    def read(self, chunk_size: int) -> bytes
