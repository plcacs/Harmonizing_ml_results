from typing import Tuple, Union, Optional, Dict
import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from pydantic.v1.typing import CallableGenerator, ReprArgs
from pydantic.v1.utils import Representation
from pydantic.v1.errors import ColorError

ColorTuple = Tuple[int, int, int, Optional[float]]
ColorType = Union[ColorTuple, str]
HslColorTuple = Tuple[float, float, float, Optional[float]]

class RGBA:
    __slots__ = ('r', 'g', 'b', 'alpha', '_tuple')

    def __init__(self, r: int, g: int, b: int, alpha: Optional[float]) -> None:
        ...

    def __getitem__(self, item: int) -> int:
        ...

r_hex_short = re.compile(r'\\s*(?:#|0x)?([0-9a-f])([0-9a-f])([0-9a-f])([0-9a-f])?\\s*')
r_hex_long = re.compile(r'\\s*(?:#|0x)?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})?\\s*')
_r_255 = re.compile(r'(\\d{1,3}(?:\\.\\d+)?)')
_r_comma = re.compile(r'\\s*,\\s*')
r_rgb = f'\\s*rgb\\(\\s*{_r_255}{_r_comma}{_r_255}{_r_comma}{_r_255}\\)\\s*'
r_rgba = f'\\s*rgba\\(\\s*{_r_255}{_r_comma}{_r_255}{_r_comma}{_r_255}{_r_comma}{_r_alpha}\\)\\s*'
_r_h = re.compile(r'(-?\\d+(?:\\.\\d+)?|-?\\.\\d+)(deg|rad|turn)?')
_r_sl = re.compile(r'(\\d{1,3}(?:\\.\\d+)?)%')
r_hsl = f'\\s*hsl\\(\\s*{_r_h}{_r_comma}{_r_sl}{_r_comma}{_r_sl}\\)\\s*'
r_hsla = f'\\s*hsl\\(\\s*{_r_h}{_r_comma}{_r_sl}{_r_comma}{_r_sl}{_r_alpha}\\)\\s*'

class Color(Representation):
    __slots__ = ('_original', '_rgba')

    def __init__(self, value: ColorType) -> None:
        ...

    @classmethod
    def __modify_schema__(cls, field_schema: Dict) -> None:
        field_schema.update(type='string', format='color')

    def original(self) -> ColorType:
        ...

    def as_named(self, *, fallback: bool = False) -> str:
        ...

    def as_hex(self) -> str:
        ...

    def as_rgb(self) -> str:
        ...

    def as_rgb_tuple(self, *, alpha: Optional[bool] = None) -> Tuple[int, int, int, Optional[float]]:
        ...

    def as_hsl(self) -> str:
        ...

    def as_hsl_tuple(self, *, alpha: Optional[bool] = None) -> Tuple[float, float, float, Optional[float]]:
        ...

    def _alpha_float(self) -> Optional[float]:
        ...

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        ...

    def __str__(self) -> str:
        ...

    def __repr_args__(self) -> ReprArgs:
        ...

    def __eq__(self, other: 'Color') -> bool:
        ...

    def __hash__(self) -> int:
        ...

def parse_tuple(value: Tuple[int, int, int]) -> RGBA:
    ...

def parse_str(value: str) -> RGBA:
    ...

def parse_color_value(value: str, max_val: int = 255) -> float:
    ...

def parse_float_alpha(value: str) -> Optional[float]:
    ...

def parse_hsl(h: str, h_units: str, sat: str, light: str, alpha: Optional[str] = None) -> RGBA:
    ...

COLORS_BY_NAME: Dict[str, Tuple[int, int, int]] = {...}
COLORS_BY_VALUE: Dict[Tuple[int, int, int], str] = {...}
