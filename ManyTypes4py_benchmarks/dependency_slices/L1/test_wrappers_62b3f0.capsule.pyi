from typing import Any

# === Internal dependency: faust ===
Record: Any

# === Internal dependency: faust.events ===
class Event(EventT):
    def __init__(self, app, key, value, headers, message): ...

# === Internal dependency: faust.exceptions ===
class ImproperlyConfigured(FaustError): ...

# === Internal dependency: faust.tables.wrappers ===
class WindowSet(WindowSetT[KT, VT]): ...

# === Internal dependency: faust.types ===
from .tuples import Message

# === Third-party dependency: mode.utils.mocks ===
class Mock(unittest.mock.Mock):
    ...
patch: Any

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises, yield_fixture