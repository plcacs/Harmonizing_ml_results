from typing import TYPE_CHECKING, Any, Final

class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        ...

def json_encoder_default(obj: Any) -> Any:
    ...

if TYPE_CHECKING:
    def json_bytes(obj: Any) -> bytes:
        ...
else:
    json_bytes = partial(orjson.dumps, option=orjson.OPT_NON_STR_KEYS, default=json_encoder_default)

class ExtendedJSONEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        ...

def _strip_null(obj: Any) -> Any:
    ...

def json_bytes_strip_null(data: Any) -> bytes:
    ...

def json_dumps(data: Any) -> str:
    ...

def json_dumps_sorted(data: Any) -> str:
    ...

def save_json(filename: str, data: Any, private: bool = False, *, encoder: Any = None, atomic_writes: bool = False) -> None:
    ...

def find_paths_unserializable_data(bad_data: Any, *, dump: Any = json.dumps) -> Any:
    ...
