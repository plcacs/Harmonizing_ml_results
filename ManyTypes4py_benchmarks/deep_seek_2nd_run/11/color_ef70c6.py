"""Color definitions are used as per the CSS3
[CSS Color Module Level 3](http://www.w3.org/TR/css3-color/#svg-color) specification.

A few colors have multiple names referring to the sames colors, eg. `grey` and `gray` or `aqua` and `cyan`.

In these cases the _last_ color when sorted alphabetically takes preferences,
eg. `Color((0, 255, 255)).as_named() == 'cyan'` because "cyan" comes after "aqua".

Warning: Deprecated
    The `Color` class is deprecated, use `pydantic_extra_types` instead.
    See [`pydantic-extra-types.Color`](../usage/types/extra_types/color_types.md)
    for more information.
"""
import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from pydantic_core import CoreSchema, PydanticCustomError, core_schema
from typing_extensions import deprecated
from ._internal import _repr
from ._internal._schema_generation_shared import GetJsonSchemaHandler as _GetJsonSchemaHandler
from .json_schema import JsonSchemaValue
from .warnings import PydanticDeprecatedSince20

ColorTuple = Union[Tuple[int, int, int], Tuple[int, int, int, float]]
ColorType = Union[ColorTuple, str]
HslColorTuple = Union[Tuple[float, float, float], Tuple[float, float, float, float]]

class RGBA:
    """Internal use only as a representation of a color."""
    __slots__ = ('r', 'g', 'b', 'alpha', '_tuple')

    def __init__(self, r: float, g: float, b: float, alpha: Optional[float]) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.alpha = alpha
        self._tuple: Tuple[float, float, float, Optional[float]] = (r, g, b, alpha)

    def __getitem__(self, item: int) -> Union[float, Optional[float]]:
        return self._tuple[item]

_r_255: str = '(\\d{1,3}(?:\\.\\d+)?)'
_r_comma: str = '\\s*,\\s*'
_r_alpha: str = '(\\d(?:\\.\\d+)?|\\.\\d+|\\d{1,2}%)'
_r_h: str = '(-?\\d+(?:\\.\\d+)?|-?\\.\\d+)(deg|rad|turn)?'
_r_sl: str = '(\\d{1,3}(?:\\.\\d+)?)%'
r_hex_short: str = '\\s*(?:#|0x)?([0-9a-f])([0-9a-f])([0-9a-f])([0-9a-f])?\\s*'
r_hex_long: str = '\\s*(?:#|0x)?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})?\\s*'
r_rgb: str = f'\\s*rgba?\\(\\s*{_r_255}{_r_comma}{_r_255}{_r_comma}{_r_255}(?:{_r_comma}{_r_alpha})?\\s*\\)\\s*'
r_hsl: str = f'\\s*hsla?\\(\\s*{_r_h}{_r_comma}{_r_sl}{_r_comma}{_r_sl}(?:{_r_comma}{_r_alpha})?\\s*\\)\\s*'
r_rgb_v4_style: str = f'\\s*rgba?\\(\\s*{_r_255}\\s+{_r_255}\\s+{_r_255}(?:\\s*/\\s*{_r_alpha})?\\s*\\)\\s*'
r_hsl_v4_style: str = f'\\s*hsla?\\(\\s*{_r_h}\\s+{_r_sl}\\s+{_r_sl}(?:\\s*/\\s*{_r_alpha})?\\s*\\)\\s*'
repeat_colors: set[int] = {int(c * 2, 16) for c in '0123456789abcdef'}
rads: float = 2 * math.pi

@deprecated('The `Color` class is deprecated, use `pydantic_extra_types` instead. See https://docs.pydantic.dev/latest/api/pydantic_extra_types_color/.', category=PydanticDeprecatedSince20)
class Color(_repr.Representation):
    """Represents a color."""
    __slots__ = ('_original', '_rgba')

    def __init__(self, value: ColorType) -> None:
        if isinstance(value, (tuple, list)):
            self._rgba = parse_tuple(value)
        elif isinstance(value, str):
            self._rgba = parse_str(value)
        elif isinstance(value, Color):
            self._rgba = value._rgba
            value = value._original
        else:
            raise PydanticCustomError('color_error', 'value is not a valid color: value must be a tuple, list or string')
        self._original = value

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: _GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema: Dict[str, Any] = {}
        field_schema.update(type='string', format='color')
        return field_schema

    def original(self) -> ColorType:
        """Original value passed to `Color`."""
        return self._original

    def as_named(self, *, fallback: bool = False) -> str:
        """Returns the name of the color if it can be found in `COLORS_BY_VALUE` dictionary,
        otherwise returns the hexadecimal representation of the color or raises `ValueError`.

        Args:
            fallback: If True, falls back to returning the hexadecimal representation of
                the color instead of raising a ValueError when no named color is found.

        Returns:
            The name of the color, or the hexadecimal representation of the color.

        Raises:
            ValueError: When no named color is found and fallback is `False`.
        """
        if self._rgba.alpha is None:
            rgb = cast(Tuple[int, int, int], self.as_rgb_tuple())
            try:
                return COLORS_BY_VALUE[rgb]
            except KeyError as e:
                if fallback:
                    return self.as_hex()
                else:
                    raise ValueError('no named color found, use fallback=True, as_hex() or as_rgb()') from e
        else:
            return self.as_hex()

    def as_hex(self) -> str:
        """Returns the hexadecimal representation of the color.

        Hex string representing the color can be 3, 4, 6, or 8 characters depending on whether the string
        a "short" representation of the color is possible and whether there's an alpha channel.

        Returns:
            The hexadecimal representation of the color.
        """
        values: List[int] = [float_to_255(c) for c in self._rgba[:3]]
        if self._rgba.alpha is not None:
            values.append(float_to_255(self._rgba.alpha))
        as_hex: str = ''.join((f'{v:02x}' for v in values))
        if all((c in repeat_colors for c in values)):
            as_hex = ''.join((as_hex[c] for c in range(0, len(as_hex), 2))
        return '#' + as_hex

    def as_rgb(self) -> str:
        """Color as an `rgb(<r>, <g>, <b>)` or `rgba(<r>, <g>, <b>, <a>)` string."""
        if self._rgba.alpha is None:
            return f'rgb({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)})'
        else:
            return f'rgba({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)}, {round(self._alpha_float(), 2)})'

    def as_rgb_tuple(self, *, alpha: Optional[bool] = None) -> Union[Tuple[int, int, int], Tuple[int, int, int, float]]:
        """Returns the color as an RGB or RGBA tuple.

        Args:
            alpha: Whether to include the alpha channel. There are three options for this input:

                - `None` (default): Include alpha only if it's set. (e.g. not `None`)
                - `True`: Always include alpha.
                - `False`: Always omit alpha.

        Returns:
            A tuple that contains the values of the red, green, and blue channels in the range 0 to 255.
                If alpha is included, it is in the range 0 to 1.
        """
        r, g, b = (float_to_255(c) for c in self._rgba[:3])
        if alpha is None:
            if self._rgba.alpha is None:
                return (r, g, b)
            else:
                return (r, g, b, self._alpha_float())
        elif alpha:
            return (r, g, b, self._alpha_float())
        else:
            return (r, g, b)

    def as_hsl(self) -> str:
        """Color as an `hsl(<h>, <s>, <l>)` or `hsl(<h>, <s>, <l>, <a>)` string."""
        if self._rgba.alpha is None:
            h, s, li = self.as_hsl_tuple(alpha=False)
            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%})'
        else:
            h, s, li, a = self.as_hsl_tuple(alpha=True)
            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%}, {round(a, 2)})'

    def as_hsl_tuple(self, *, alpha: Optional[bool] = None) -> Union[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Returns the color as an HSL or HSLA tuple.

        Args:
            alpha: Whether to include the alpha channel.

                - `None` (default): Include the alpha channel only if it's set (e.g. not `None`).
                - `True`: Always include alpha.
                - `False`: Always omit alpha.

        Returns:
            The color as a tuple of hue, saturation, lightness, and alpha (if included).
                All elements are in the range 0 to 1.

        Note:
            This is HSL as used in HTML and most other places, not HLS as used in Python's `colorsys`.
        """
        h, l, s = rgb_to_hls(self._rgba.r, self._rgba.g, self._rgba.b)
        if alpha is None:
            if self._rgba.alpha is None:
                return (h, s, l)
            else:
                return (h, s, l, self._alpha_float())
        if alpha:
            return (h, s, l, self._alpha_float())
        else:
            return (h, s, l)

    def _alpha_float(self) -> float:
        return 1 if self._rgba.alpha is None else self._rgba.alpha

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], CoreSchema]) -> CoreSchema:
        return core_schema.with_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

    @classmethod
    def _validate(cls, __input_value: Any, _: Any) -> 'Color':
        return cls(__input_value)

    def __str__(self) -> str:
        return self.as_named(fallback=True)

    def __repr_args__(self) -> List[Tuple[Optional[str], Any]]:
        return [(None, self.as_named(fallback=True))] + [('rgb', self.as_rgb_tuple())]

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Color) and self.as_rgb_tuple() == other.as_rgb_tuple()

    def __hash__(self) -> int:
        return hash(self.as_rgb_tuple())

def parse_tuple(value: Union[Tuple[Any, ...], List[Any]]) -> RGBA:
    """Parse a tuple or list to get RGBA values.

    Args:
        value: A tuple or list.

    Returns:
        An `RGBA` tuple parsed from the input tuple.

    Raises:
        PydanticCustomError: If tuple is not valid.
    """
    if len(value) == 3:
        r, g, b = (parse_color_value(v) for v in value)
        return RGBA(r, g, b, None)
    elif len(value) == 4:
        r, g, b = (parse_color_value(v) for v in value[:3])
        return RGBA(r, g, b, parse_float_alpha(value[3]))
    else:
        raise PydanticCustomError('color_error', 'value is not a valid color: tuples must have length 3 or 4')

def parse_str(value: str) -> RGBA:
    """Parse a string representing a color to an RGBA tuple.

    Possible formats for the input string include:

    * named color, see `COLORS_BY_NAME`
    * hex short eg. `<prefix>fff` (prefix can be `#`, `0x` or nothing)
    * hex long eg. `<prefix>ffffff` (prefix can be `#`, `0x` or nothing)
    * `rgb(<r>, <g>, <b>)`
    * `rgba(<r>, <g>, <b>, <a>)`

    Args:
        value: A string representing a color.

    Returns:
        An `RGBA` tuple parsed from the input string.

    Raises:
        ValueError: If the input string cannot be parsed to an RGBA tuple.
    """
    value_lower = value.lower()
    try:
        r, g, b = COLORS_BY_NAME[value_lower]
    except KeyError:
        pass
    else:
        return ints_to_rgba(r, g, b, None)
    m = re.fullmatch(r_hex_short, value_lower)
    if m:
        *rgb, a = m.groups()
        r, g, b = (int(v * 2, 16) for v in rgb)
        if a:
            alpha = int(a * 2, 16) / 255
        else:
            alpha = None
        return ints_to_rgba(r, g, b, alpha)
    m = re.fullmatch(r_hex_long, value_lower)
    if m:
        *rgb, a = m.groups()
        r, g, b = (int(v, 16) for v in rgb)
        if a:
            alpha = int(a, 16) / 255
        else:
            alpha = None
        return ints_to_rgba(r, g, b, alpha)
    m = re.fullmatch(r_rgb, value_lower) or re.fullmatch(r_rgb_v4_style, value_lower)
    if m:
        return ints_to_rgba(*m.groups())
    m = re.fullmatch(r_hsl, value_lower) or re.fullmatch(r_hsl_v4_style, value_lower)
    if m:
        return parse_hsl(*m.groups())
    raise PydanticCustomError('color_error', 'value is not a valid color: string not recognised as a valid color')

def ints_to_rgba(r: Union[int, str], g: Union[int, str], b: Union[int, str], alpha: Optional[Union[float, str]]) -> RGBA:
    """Converts integer or string values for RGB color and an optional alpha value to an `RGBA` object.

    Args:
        r: An integer or string representing the red color value.
        g: An integer or string representing the green color value.
        b: An integer or string representing the blue color value.
        alpha: A float representing the alpha value. Defaults to None.

    Returns:
        An instance of the `RGBA` class with the corresponding color and alpha values.
    """
    return RGBA(parse_color_value(r), parse_color_value(g), parse_color_value(b), parse_float_alpha(alpha))

def parse_color_value(value: Union[int, str], max_val: int = 255) -> float:
    """Parse the color value provided and return a number between 0 and 1.

    Args:
        value: An integer or string color value.
        max_val: Maximum range value. Defaults to 255.

    Raises:
        PydanticCustomError: If the value is not a valid color.

    Returns:
        A number between 0 and 1.
    """
    try:
        color = float(value)
    except ValueError:
        raise PydanticCustomError('color_error', 'value is not a valid color: color values must be a valid number')
    if 0 <= color <= max_val:
        return color / max_val
    else:
        raise PydanticCustomError('color_error', 'value is not a valid color: color values must be in the range 0 to {max_val}', {'max_val': max_val})

def parse_float_alpha(value: Optional[Union[float, str]]) -> Optional[float]:
    """Parse an alpha value checking it's a valid float in the range 0 to 1.

    Args:
        value: The input value to parse.

    Returns:
        The parsed value as a float, or `None` if the value was None or equal 1.

    Raises:
        PydanticCustomError: If the input value cannot be successfully parsed as a float in the expected range.
    """
    if value is None:
        return None
    try:
        if isinstance