"""
Color definitions are used as per CSS3 specification:
http://www.w3.org/TR/css3-color/#svg-color

A few colors have multiple names referring to the sames colors, eg. `grey` and `gray` or `aqua` and `cyan`.

In these cases the LAST color when sorted alphabetically takes preferences,
eg. Color((0, 255, 255)).as_named() == 'cyan' because "cyan" comes after "aqua".
"""
import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast, List, Pattern
from pydantic.v1.errors import ColorError
from pydantic.v1.utils import Representation, almost_equal_floats

if TYPE_CHECKING:
    from pydantic.v1.typing import CallableGenerator, ReprArgs

ColorTuple = Union[Tuple[int, int, int], Tuple[int, int, int, float]]
ColorType = Union[ColorTuple, str]
HslColorTuple = Union[Tuple[float, float, float], Tuple[float, float, float, float]]

class RGBA:
    """
    Internal use only as a representation of a color.
    """
    __slots__ = ('r', 'g', 'b', 'alpha', '_tuple')

    def __init__(self, r: float, g: float, b: float, alpha: Optional[float]) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.alpha = alpha
        self._tuple: Tuple[float, float, float, Optional[float]] = (r, g, b, alpha)

    def __getitem__(self, item: int) -> Union[float, Optional[float]]:
        return self._tuple[item]

r_hex_short: str = '\\s*(?:#|0x)?([0-9a-f])([0-9a-f])([0-9a-f])([0-9a-f])?\\s*'
r_hex_long: str = '\\s*(?:#|0x)?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})?\\s*'
_r_255: str = '(\\d{1,3}(?:\\.\\d+)?)'
_r_comma: str = '\\s*,\\s*'
r_rgb: str = f'\\s*rgb\\(\\s*{_r_255}{_r_comma}{_r_255}{_r_comma}{_r_255}\\)\\s*'
_r_alpha: str = '(\\d(?:\\.\\d+)?|\\.\\d+|\\d{1,2}%)'
r_rgba: str = f'\\s*rgba\\(\\s*{_r_255}{_r_comma}{_r_255}{_r_comma}{_r_255}{_r_comma}{_r_alpha}\\s*\\)\\s*'
_r_h: str = '(-?\\d+(?:\\.\\d+)?|-?\\.\\d+)(deg|rad|turn)?'
_r_sl: str = '(\\d{1,3}(?:\\.\\d+)?)%'
r_hsl: str = f'\\s*hsl\\(\\s*{_r_h}{_r_comma}{_r_sl}{_r_comma}{_r_sl}\\s*\\)\\s*'
r_hsla: str = f'\\s*hsl\\(\\s*{_r_h}{_r_comma}{_r_sl}{_r_comma}{_r_sl}{_r_comma}{_r_alpha}\\s*\\)\\s*'
repeat_colors: set = {int(c * 2, 16) for c in '0123456789abcdef'}
rads: float = 2 * math.pi

class Color(Representation):
    __slots__ = ('_original', '_rgba')

    def __init__(self, value: ColorType) -> None:
        if isinstance(value, (tuple, list)):
            self._rgba: RGBA = parse_tuple(value)
        elif isinstance(value, str):
            self._rgba = parse_str(value)
        elif isinstance(value, Color):
            self._rgba = value._rgba
            value = value._original
        else:
            raise ColorError(reason='value must be a tuple, list or string')
        self._original: ColorType = value

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='color')

    def original(self) -> ColorType:
        """
        Original value passed to Color
        """
        return self._original

    def as_named(self, *, fallback: bool = False) -> str:
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
        """
        Hex string representing the color can be 3, 4, 6 or 8 characters depending on whether the string
        a "short" representation of the color is possible and whether there's an alpha channel.
        """
        values: List[int] = [float_to_255(c) for c in self._rgba[:3]]
        if self._rgba.alpha is not None:
            values.append(float_to_255(self._rgba.alpha))
        as_hex: str = ''.join((f'{v:02x}' for v in values))
        if all((c in repeat_colors for c in values)):
            as_hex = ''.join((as_hex[c] for c in range(0, len(as_hex), 2)))
        return '#' + as_hex

    def as_rgb(self) -> str:
        """
        Color as an rgb(<r>, <g>, <b>) or rgba(<r>, <g>, <b>, <a>) string.
        """
        if self._rgba.alpha is None:
            return f'rgb({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)})'
        else:
            return f'rgba({float_to_255(self._rgba.r)}, {float_to_255(self._rgba.g)}, {float_to_255(self._rgba.b)}, {round(self._alpha_float(), 2)})'

    def as_rgb_tuple(self, *, alpha: Optional[bool] = None) -> Union[Tuple[int, int, int], Tuple[int, int, int, float]]:
        """
        Color as an RGB or RGBA tuple; red, green and blue are in the range 0 to 255, alpha if included is
        in the range 0 to 1.

        :param alpha: whether to include the alpha channel, options are
          None - (default) include alpha only if it's set (e.g. not None)
          True - always include alpha,
          False - always omit alpha,
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
        """
        Color as an hsl(<h>, <s>, <l>) or hsl(<h>, <s>, <l>, <a>) string.
        """
        if self._rgba.alpha is None:
            h, s, li = self.as_hsl_tuple(alpha=False)
            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%})'
        else:
            h, s, li, a = self.as_hsl_tuple(alpha=True)
            return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%}, {round(a, 2)})'

    def as_hsl_tuple(self, *, alpha: Optional[bool] = None) -> Union[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Color as an HSL or HSLA tuple, e.g. hue, saturation, lightness and optionally alpha; all elements are in
        the range 0 to 1.

        NOTE: this is HSL as used in HTML and most other places, not HLS as used in python's colorsys.

        :param alpha: whether to include the alpha channel, options are
          None - (default) include alpha only if it's set (e.g. not None)
          True - always include alpha,
          False - always omit alpha,
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
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls

    def __str__(self) -> str:
        return self.as_named(fallback=True)

    def __repr_args__(self) -> 'ReprArgs':
        return [(None, self.as_named(fallback=True))] + [('rgb', self.as_rgb_tuple())]

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Color) and self.as_rgb_tuple() == other.as_rgb_tuple()

    def __hash__(self) -> int:
        return hash(self.as_rgb_tuple())

def parse_tuple(value: Union[Tuple[Any, ...], List[Any]]) -> RGBA:
    """
    Parse a tuple or list as a color.
    """
    if len(value) == 3:
        r, g, b = (parse_color_value(v) for v in value)
        return RGBA(r, g, b, None)
    elif len(value) == 4:
        r, g, b = (parse_color_value(v) for v in value[:3])
        return RGBA(r, g, b, parse_float_alpha(value[3]))
    else:
        raise ColorError(reason='tuples must have length 3 or 4')

def parse_str(value: str) -> RGBA:
    """
    Parse a string to an RGBA tuple, trying the following formats (in this order):
    * named color, see COLORS_BY_NAME below
    * hex short eg. `<prefix>fff` (prefix can be `#`, `0x` or nothing)
    * hex long eg. `<prefix>ffffff` (prefix can be `#`, `0x` or nothing)
    * `rgb(<r>, <g>, <b>) `
    * `rgba(<r>, <g>, <b>, <a>)`
    """
    value_lower: str = value.lower()
    try:
        r, g, b = COLORS_BY_NAME[value_lower]
    except KeyError:
        pass
    else:
        return ints_to_rgba(r, g, b, None)
    m: Optional[Pattern] = re.fullmatch(r_hex_short, value_lower)
    if m:
        *rgb, a = m.groups()
        r, g, b = (int(v * 2, 16) for v in rgb)
        if a:
            alpha: Optional[float] = int(a * 2, 16) / 255
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
    m = re.fullmatch(r_rgb, value_lower)
    if m:
        return ints_to_rgba(*m.groups(), None)
    m = re.fullmatch(r_rgba, value_lower)
    if m:
        return ints_to_rgba(*m.groups())
    m = re.fullmatch(r_hsl, value_lower)
    if m:
        h, h_units, s, l_ = m.groups()
        return parse_hsl(h, h_units, s, l_)
    m = re.fullmatch(r_hsla, value_lower)
    if m:
        h, h_units, s, l_, a = m.groups()
        return parse_hsl(h, h_units, s, l_, parse_float_alpha(a))
    raise ColorError(reason='string not recognised as a valid color')

def ints_to_rgba(r: Union[str, int], g: Union[str, int], b: Union[str, int], alpha: Optional[Union[str, float]]) -> RGBA:
    return RGBA(parse_color_value(r), parse_color_value(g), parse_color_value(b), parse_float_alpha(alpha))

def parse_color_value(value: Union[str, int, float], max_val: int = 255) -> float:
    """
    Parse a value checking it's a valid int in the range 0 to max_val and divide by max_val to give a number
    in the range 0 to 1
    """
    try:
        color: float = float(value)
    except ValueError:
        raise ColorError(reason='color values must be a valid number')
    if 0 <= color <= max_val:
        return color / max_val
    else:
        raise ColorError(reason=f'color values must be in the range 0 to {max_val}')

def parse_float_alpha(value: Optional[Union[str, float]]) -> Optional[float]:
    """
    Parse a value checking it's a valid float in the range 0 to 1
    """
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.endswith('%'):
            alpha: float = float(value[:-1]) / 100
        else:
            alpha = float(value)
    except ValueError:
        raise ColorError(reason='alpha values must be a valid float')
    if almost_equal_floats(alpha, 1):
        return None
    elif 0 <= alpha <= 1:
        return alpha
    else:
        raise ColorError(reason='alpha values must be in the range 0 to 1')

def parse_hsl(h: str, h_units: Optional[str], sat: str, light: str, alpha: Optional[float] = None) -> RGBA:
    """
    Parse raw hue, saturation, lightness and alpha values and convert to RGBA.
    """
    s_value, l_value = (parse_color_value(sat, 100), parse_color_value(light, 100))
    h_value: float = float(h)
    if h_units in {None, 'deg'}:
        h_value = h_value % 360 / 360
    elif h_units == 'rad':
        h_value = h_value % rads / rads
    else:
        h_value = h_value % 1
    r, g, b = hls_to_rgb(h_value, l_value, s_value)
    return RGBA(r, g, b, alpha)

def float_to_255(c: float) -> int:
    return int(round(c * 255))

COLORS_BY_NAME: Dict[str, Tuple[int, int, int]] = {'aliceblue': (240, 248, 255), 'antiquewhite': (250, 235, 215), 'aqua': (0, 255, 255), 'aquamarine': (127, 255, 212), 'azure': (240, 255, 255), 'beige': (245, 245, 220), 'bisque': (255, 228, 196), 'black': (0, 0, 0), 'blanchedalmond': (255, 235, 205), 'blue': (0, 0, 255), 'blueviolet': (138, 43, 226), 'brown': (165, 42, 42), 'burlywood': (222, 184, 135), 'cadetblue': (95, 158, 160), 'chartreuse': (127, 255, 0), 'chocolate': (210, 105, 30), 'coral': (255, 127, 80), 'cornflowerblue': (100, 149, 237), 'cornsilk': (255, 248, 220), 'crimson': (220, 20, 60), 'cyan': (0, 255, 255), 'darkblue': (0, 0, 139), 'darkcyan': (0, 139, 139), 'darkgoldenrod': (184, 134, 11), 'darkgray': (169, 169, 169), 'darkgreen': (0, 100, 0), 'darkgrey': (169, 169, 169), 'darkkhaki': (189, 183, 107), 'darkmagenta': (139, 0, 139), 'darkolivegreen': (85, 107, 47), 'darkorange': (255, 140, 0