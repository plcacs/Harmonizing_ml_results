"""
Stub file for 'color_334856' module
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic.v1.typing import CallableGenerator, ReprArgs

r_hex_short = str
r_hex_long = str
_r_255 = str
_r_comma = str
r_rgb = str
_r_alpha = str
r_rgba = str
_r_h = str
_r_sl = str
r_hsl = str
r_hsla = str
repeat_colors = set[int]
rads = float

class RGBA:
    __slots__ = ('r', 'g', 'b', 'alpha', '_tuple')
    r: float
    g: float
    b: float
    alpha: Optional[float]
    _tuple: Tuple[float, float, float, Optional[float]]
    
    def __init__(self, r: float, g: float, b: float, alpha: Optional[float]) -> None:
        ...
        
    def __getitem__(self, item: int) -> float:
        ...

class Color:
    __slots__ = ('_original', '_rgba')
    _original: Any
    _rgba: RGBA
    
    def __init__(self, value: Union[Tuple[int, int, int], Tuple[int, int, int, float], str, Color]) -> None:
        ...
        
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        ...
        
    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        ...
        
    def original(self) -> Any:
        ...
        
    def as_named(self, *, fallback: bool = False) -> str:
        ...
        
    def as_hex(self) -> str:
        ...
        
    def as_rgb(self) -> str:
        ...
        
    def as_rgb_tuple(self, *, alpha: Optional[bool] = None) -> Union[Tuple[int, int, int], Tuple[int, int, int, float]]:
        ...
        
    def as_hsl(self) -> str:
        ...
        
    def as_hsl_tuple(self, *, alpha: Optional[bool] = None) -> Union[Tuple[float, float, float], Tuple[float, float, float, float]]:
        ...
        
    def _alpha_float(self) -> float:
        ...
        
    def __str__(self) -> str:
        ...
        
    def __repr_args__(self) -> ReprArgs:
        ...
        
    def __eq__(self, other: Any) -> bool:
        ...
        
    def __hash__(self) -> int:
        ...

def parse_tuple(value: Union[Tuple[Any, Any, Any], List[Any]]) -> RGBA:
    ...

def parse_str(value: str) -> RGBA:
    ...

def ints_to_rgba(r: int, g: int, b: int, alpha: Optional[int]) -> RGBA:
    ...

def parse_color_value(value: Union[str, float, int], max_val: int = 255) -> float:
    ...

def parse_float_alpha(value: Optional[Union[str, float, int]]) -> Optional[float]:
    ...

def parse_hsl(h: str, h_units: Optional[str], sat: str, light: str, alpha: Optional[float] = None) -> RGBA:
    ...

def float_to_255(c: float) -> int:
    ...

COLORS_BY_NAME: Dict[str, Tuple[int, int, int]]
COLORS_BY_VALUE: Dict[Tuple[int, int, int], str]