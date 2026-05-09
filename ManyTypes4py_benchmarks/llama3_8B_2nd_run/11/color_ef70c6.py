from typing import Union, Callable, Optional, Any, Tuple
import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from pydantic_core import CoreSchema, PydanticCustomError, core_schema
from typing_extensions import deprecated
from ._internal import _repr
from ._internal._schema_generation_shared import GetJsonSchemaHandler as _GetJsonSchemaHandler
from .json_schema import JsonSchemaValue
from .warnings import PydanticDeprecatedSince20

ColorTuple = Union[tuple[int, int, int], tuple[int, int, int, float]]
ColorType = Union[ColorTuple, str]
HslColorTuple = Union[tuple[float, float, float], tuple[float, float, float, float]]

class RGBA:
    """Internal use only as a representation of a color."""
    __slots__ = ('r', 'g', 'b', 'alpha', '_tuple')

    def __init__(self, r: float, g: float, b: float, alpha: Optional[float] = None):
        self.r = r
        self.g = g
        self.b = b
        self.alpha = alpha
        self._tuple = (r, g, b, alpha)

    def __getitem__(self, item: int) -> float:
        return self._tuple[item]

_r_255 = re.compile(r'(\d{1,3}(?:\.\d+)?)')
_r_comma = re.compile(r'\s*,\s*')
_r_alpha = re.compile(r'(\d(?:\.\d+)?|\.?\d+)(?:%|%)?')
r_hex_short = re.compile(r'\s*(?:#|0x)?([0-9a-f])([0-9a-f])([0-9a-f])([0-9a-f])?\s*')
r_hex_long = re.compile(r'\s*(?:#|0x)?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})?\s*')
r_rgb = re.compile(f'\s*rgba?\s*\(\s*{_r_255}{_r_comma}{_r_255}{_r_comma}{_r_255}(?:{_r_comma}{_r_alpha})?\s*\)\s*')
r_hsl = re.compile(f'\s*hsla?\s*\(\s*{_r_h}{_r_comma}{_r_sl}{_r_comma}{_r_sl}(?:{_r_comma}{_r_alpha})?\s*\)\s*')
r_rgb_v4_style = re.compile(f'\s*rgba?\s*\(\s*{_r_255}\s+{_r_255}\s+{_r_255}(?:\s*/\s*{_r_alpha})?\s*\)\s*')
r_hsl_v4_style = re.compile(f'\s*hsla?\s*\(\s*{_r_h}\s+{_r_sl}\s+{_r_sl}(?:\s*/\s*{_r_alpha})?\s*\)\s*')

@deprecated('The `Color` class is deprecated, use `pydantic_extra_types` instead. See https://docs.pydantic.dev/latest/api/pydantic_extra_types_color/.', category=PydanticDeprecatedSince20)
class Color(_repr.Representation):
    """Represents a color."""
    __slots__ = ('_original', '_rgba')

    def __init__(self, value: ColorType):
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
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: Callable[[Any], Any]) -> dict:
        field_schema = {}
        field_schema.update(type='string', format='color')
        return field_schema

    def original(self) -> Any:
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
        values = [float_to_255(c) for c in self._rgba[:3]]
        if self._rgba.alpha is not None:
            values.append(float_to_255(self._rgba.alpha))
        as_hex = ''.join((f'{v:02x}' for v in values))
        if all((c in repeat_colors for c in values)):
            as_hex = ''.join((as_hex[c] for c in range(0, len(as_hex), 2)))
        return '#' + as_hex

    def as_rgb(self) -> str:
        """Color as an `rgb(<r>, <g>, <b>)` or `rgba(<r>, <g>, <b>, <a>)` string."""
        if self._rgba.alpha is None:
            return f'rgb({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)})'
        else:
            return f'rgba({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)}, {round(self._alpha_float(), 2)})'

    def as_rgb_tuple(self, *, alpha: Optional[bool] = None) -> Tuple[int, int, int]:
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
        if alpha:
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

    def as_hsl_tuple(self, *, alpha: Optional[bool] = None) -> Tuple[float, float, float]:
        """Parse raw hue, saturation, lightness, and alpha values and convert to RGBA.

        Args:
            h: The hue value.
            h_units: The unit for hue value.
            sat: The saturation value.
            light: The lightness value.
            alpha: Alpha value.

        Returns:
            An instance of `RGBA`.
        """
        s_value, l_value = (parse_color_value(sat, 100), parse_color_value(light, 100))
        h_value = float(h)
        if h_units in {None, 'deg'}:
            h_value = h_value % 360 / 360
        elif h_units == 'rad':
            h_value = h_value % rads / rads
        else:
            h_value = h_value % 1
        r, g, b = hls_to_rgb(h_value, l_value, s_value)
        return RGBA(r, g, b, parse_float_alpha(alpha))

    def _alpha_float(self) -> Optional[float]:
        return 1 if self._rgba.alpha is