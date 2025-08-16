from typing import List, Tuple

class ErrorParams(namedtuple):
    code: str
    short_desc: str
    context: str

class Error:
    code: str
    short_desc: str
    context: str
    parameters: Tuple
    definition: str
    explanation: str

    def __init__(self, code: str, short_desc: str, context: str, *parameters: Tuple):
        ...

    def set_context(self, definition: str, explanation: str):
        ...

    @property
    def message(self) -> str:
        ...

    @property
    def lines(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __lt__(self, other: 'Error') -> bool:
        ...

class ErrorRegistry:
    groups: List

    class ErrorGroup:
        prefix: str
        name: str
        errors: List

        def __init__(self, prefix: str, name: str):
            ...

        def create_error(self, error_code: str, error_desc: str, error_context: str = None) -> Error:
            ...

    @classmethod
    def create_group(cls, prefix: str, name: str) -> 'ErrorGroup':
        ...

    @classmethod
    def get_error_codes(cls) -> List[str]:
        ...

    @classmethod
    def to_rst(cls) -> str:
        ...

D1xx: ErrorRegistry.ErrorGroup
D100: Error
D101: Error
D102: Error
D103: Error
D104: Error
D105: Error
D106: Error
D2xx: ErrorRegistry.ErrorGroup
D200: Error
D201: Error
D202: Error
D203: Error
D204: Error
D205: Error
D206: Error
D207: Error
D208: Error
D209: Error
D210: Error
D211: Error
D212: Error
D213: Error
D214: Error
D215: Error
D3xx: ErrorRegistry.ErrorGroup
D300: Error
D301: Error
D302: Error
D4xx: ErrorRegistry.ErrorGroup
D400: Error
D401: Error
D401b: Error
D402: Error
D403: Error
D404: Error
D405: Error
D406: Error
D407: Error
D408: Error
D409: Error
D410: Error
D411: Error
D412: Error
D413: Error
D414: Error

class AttrDict(dict):
    def __getattr__(self, item: str) -> str:
        ...
