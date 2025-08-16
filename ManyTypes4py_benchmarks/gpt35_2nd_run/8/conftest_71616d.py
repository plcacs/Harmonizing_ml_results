from typing import Any, NamedTuple

class TimeMarks(NamedTuple):
    time: Any = None
    monotonic: Any = None

class SessionMarker(NamedTuple):
    status_code: Any = HTTPStatus.OK
    text: Any = b''
    json: Any = None
    json_iterator: Any = None
    max_failures: Any = None
