from __future__ import annotations
import colorsys
import math
from typing import NamedTuple, Tuple
import attr

class RGBColor(NamedTuple):
    """RGB hex values."""
    r: int
    g: int
    b: int

COLORS: dict[str, RGBColor] = {
    # ...
}

@attr.s()
class XYPoint:
    """Represents a CIE 1931 XY coordinate pair."""
    x: float = attr.ib()
    y: float = attr.ib()

@attr.s()
class GamutType:
    """Represents the Gamut of a light."""
    red: XYPoint = attr.ib()
    green: XYPoint = attr.ib()
    blue: XYPoint = attr.ib()

def color_name_to_rgb(color_name: str) -> RGBColor:
    """Convert color name to RGB hex value."""
    hex_value = COLORS.get(color_name.replace(' ', '').lower())
    if not hex_value:
        raise ValueError('Unknown color')
    return hex_value

def color_RGB_to_xy(r: int, g: int, b: int, Gamut: GamutType | None = None) -> Tuple[float, float]:
    """Convert from RGB color to XY color."""
    return color_RGB_to_xy_brightness(r, g, b, Gamut)[:2]

def color_RGB_to_xy_brightness(r: int, g: int, b: int, Gamut: GamutType | None = None) -> Tuple[float, float, int]:
    """Convert from RGB color to XY color."""
    if r + g + b == 0:
        return (0.0, 0.0, 0)
    # ...

def color_xy_to_RGB(x: float, y: float, Gamut: GamutType | None = None) -> Tuple[int, int, int]:
    """Convert from XY to a normalized RGB."""
    return color_xy_brightness_to_RGB(x, y, 255, Gamut)

def color_xy_brightness_to_RGB(x: float, y: float, brightness: int, Gamut: GamutType | None = None) -> Tuple[int, int, int]:
    """Convert from XYZ to RGB."""
    if Gamut and (not check_point_in_lamps_reach((x, y), Gamut)):
        xy_closest = get_closest_point_to_point((x, y), Gamut)
        x = xy_closest[0]
        y = xy_closest[1]
    # ...

def color_hsb_to_RGB(h: float, s: float, b: float) -> Tuple[int, int, int]:
    """Convert a hsb into its rgb representation."""
    if s == 0.0:
        fV = int(b * 255)
        return (fV, fV, fV)
    # ...

def color_RGB_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert an rgb color to its hsv representation.

    Hue is scaled 0-360
    Sat is scaled 0-100
    Val is scaled 0-100
    """
    fHSV = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return (round(fHSV[0] * 360, 3), round(fHSV[1] * 100, 3), round(fHSV[2] * 100, 3))

def color_RGB_to_hs(r: int, g: int, b: int) -> Tuple[float, float]:
    """Convert an rgb color to its hs representation."""
    return color_RGB_to_hsv(r, g, b)[:2]

def color_hsv_to_RGB(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert an hsv color into its rgb representation.

    Hue is scaled 0-360
    Sat is scaled 0-100
    Val is scaled 0-100
    """
    fRGB = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)
    return (round(fRGB[0] * 255), round(fRGB[1] * 255), round(fRGB[2] * 255))

def color_hs_to_RGB(h: float, s: float) -> Tuple[int, int, int]:
    """Convert an hsv color into its rgb representation."""
    return color_hsv_to_RGB(h, s, 100)

def color_xy_to_hs(x: float, y: float, Gamut: GamutType | None = None) -> Tuple[float, float]:
    """Convert an xy color to its hs representation."""
    h, s, _ = color_RGB_to_hsv(*color_xy_to_RGB(x, y, Gamut))
    return (h, s)

def color_hs_to_xy(h: float, s: float, Gamut: GamutType | None = None) -> Tuple[float, float]:
    """Convert an hs color to its xy representation."""
    return color_RGB_to_xy(*color_hs_to_RGB(h, s), Gamut)

def match_max_scale(input_colors: Tuple[int, ...], output_colors: Tuple[int, ...]) -> Tuple[int, ...]:
    """Match the maximum value of the output to the input."""
    max_in = max(input_colors)
    max_out = max(output_colors)
    if max_out == 0:
        factor = 0.0
    else:
        factor = max_in / max_out
    return tuple((int(round(i * factor)) for i in output_colors))

def color_rgb_to_rgbw(r: int, g: int, b: int) -> Tuple[int, int, int, int]:
    """Convert an rgb color to an rgbw representation."""
    w = min(r, g, b)
    rgbw = (r - w, g - w, b - w, w)
    return match_max_scale((r, g, b), rgbw)

def color_rgbw_to_rgb(r: int, g: int, b: int, w: int) -> Tuple[int, int, int]:
    """Convert an rgbw color to an rgb representation."""
    rgb = (r + w, g + w, b + w)
    return match_max_scale((r, g, b, w), rgb)

def color_rgb_to_rgbww(r: int, g: int, b: int, min_kelvin: int, max_kelvin: int) -> Tuple[int, int, int, int, int]:
    """Convert an rgb color to an rgbww representation."""
    max_mireds = color_temperature_kelvin_to_mired(min_kelvin)
    min_mireds = color_temperature_kelvin_to_mired(max_kelvin)
    mired_range = max_mireds - min_mireds
    mired_midpoint = min_mireds + mired_range / 2
    color_temp_kelvin = color_temperature_mired_to_kelvin(mired_midpoint)
    w_r, w_g, w_b = color_temperature_to_rgb(color_temp_kelvin)
    white_level = min(r / w_r if w_r else 0, g / w_g if w_g else 0, b / w_b if w_b else 0)
    rgb = (r - w_r * white_level, g - w_g * white_level, b - w_b * white_level)
    rgbww = (*rgb, round(white_level * 255), round(white_level * 255))
    return match_max_scale((r, g, b), rgbww)

def color_rgbww_to_rgb(r: int, g: int, b: int, cw: int, ww: int, min_kelvin: int, max_kelvin: int) -> Tuple[int, int, int]:
    """Convert an rgbww color to an rgb representation."""
    max_mireds = color_temperature_kelvin_to_mired(min_kelvin)
    min_mireds = color_temperature_kelvin_to_mired(max_kelvin)
    mired_range = max_mireds - min_mireds
    try:
        ct_ratio = ww / (cw + ww)
    except ZeroDivisionError:
        ct_ratio = 0.5
    color_temp_mired = min_mireds + ct_ratio * mired_range
    if color_temp_mired:
        color_temp_kelvin = color_temperature_mired_to_kelvin(color_temp_mired)
    else:
        color_temp_kelvin = 0
    w_r, w_g, w_b = color_temperature_to_rgb(color_temp_kelvin)
    white_level = max(cw, ww) / 255
    rgb = (r + w_r * white_level, g + w_g * white_level, b + w_b * white_level)
    return match_max_scale((r, g, b, cw, ww), rgb)

def color_rgb_to_hex(r: int, g: int, b: int) -> str:
    """Return a RGB color from a hex color string."""
    return f'{round(r):02x}{round(g):02x}{round(b):02x}'

def rgb_hex_to_rgb_list(hex_string: str) -> Tuple[int, int, int]:
    """Return an RGB color value list from a hex color string."""
    return tuple(int(hex_string[i:i + len(hex_string) // 3], 16) for i in range(0, len(hex_string), len(hex_string) // 3))

def color_temperature_to_hs(color_temperature_kelvin: int) -> Tuple[float, float]:
    """Return an hs color from a color temperature in Kelvin."""
    return color_RGB_to_hs(*color_temperature_to_rgb(color_temperature_kelvin))

def color_temperature_to_rgb(color_temperature_kelvin: int) -> Tuple[int, int, int]:
    """Return an RGB color from a color temperature in Kelvin.

    This is a rough approximation based on the formula provided by T. Helland
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    if color_temperature_kelvin < 1000:
        color_temperature_kelvin = 1000
    elif color_temperature_kelvin > 40000:
        color_temperature_kelvin = 40000
    tmp_internal = color_temperature_kelvin / 100.0
    red = _get_red(tmp_internal)
    green = _get_green(tmp_internal)
    blue = _get_blue(tmp_internal)
    return (red, green, blue)

def color_temperature_to_rgbww(temperature: int, brightness: int, min_kelvin: int, max_kelvin: int) -> Tuple[int, int, int, int, int]:
    """Convert color temperature in kelvin to rgbcw.

    Returns a (r, g, b, cw, ww) tuple.
    """
    max_mireds = color_temperature_kelvin_to_mired(min_kelvin)
    min_mireds = color_temperature_kelvin_to_mired(max_kelvin)
    temperature = color_temperature_kelvin_to_mired(temperature)
    mired_range = max_mireds - min_mireds
    cold = (max_mireds - temperature) / mired_range * brightness
    warm = brightness - cold
    return (0, 0, 0, round(cold), round(warm))

def rgbww_to_color_temperature(rgbww: Tuple[int, int, int, int, int], min_kelvin: int, max_kelvin: int) -> Tuple[int, int]:
    """Convert rgbcw to color temperature in kelvin.

    Returns a tuple (color_temperature, brightness).
    """
    _, _, _, cold, warm = rgbww
    return _white_levels_to_color_temperature(cold, warm, min_kelvin, max_kelvin)

def _white_levels_to_color_temperature(cold: int, warm: int, min_kelvin: int, max_kelvin: int) -> Tuple[int, int]:
    """Convert whites to color temperature in kelvin.

    Returns a tuple (color_temperature, brightness).
    """
    max_mireds = color_temperature_kelvin_to_mired(min_kelvin)
    min_mireds = color_temperature_kelvin_to_mired(max_kelvin)
    brightness = warm / 255 + cold / 255
    if brightness == 0:
        return (min_kelvin, 0)
    return (round(color_temperature_mired_to_kelvin(cold / 255 / brightness * (min_mireds - max_mireds) + max_mireds)), min(255, round(brightness * 255)))

def color_xy_to_temperature(x: float, y: float) -> int:
    """Convert an xy color to a color temperature in Kelvin.

    Uses McCamy's approximation (https://doi.org/10.1002/col.5080170211),
    close enough for uses between 2000 K and 10000 K.
    """
    n = (x - 0.332) / (0.1858 - y)
    CCT = 437 * n ** 3 + 3601 * n ** 2 + 6861 * n + 5517
    return int(CCT)

def _clamp(color_component: float, minimum: int = 0, maximum: int = 255) -> float:
    """Clamp the given color component value between the given min and max values.

    The range defined by the minimum and maximum values is inclusive, i.e. given a
    color_component of 0 and a minimum of 10, the returned value is 10.
    """
    color_component_out = max(color_component, minimum)
    return min(color_component_out, maximum)

def _get_red(temperature: float) -> float:
    """Get the red component of the temperature in RGB space."""
    if temperature <= 66:
        return 255
    tmp_red = 329.698727446 * math.pow(temperature - 60, -0.1332047592)
    return _clamp(tmp_red)

def _get_green(temperature: float) -> float:
    """Get the green component of the given color temp in RGB space."""
    if temperature <= 66:
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
    else:
        green = 288.1221695283 * math.pow(temperature - 60, -0.0755148492)
    return _clamp(green)

def _get_blue(temperature: float) -> float:
    """Get the blue component of the given color temperature in RGB space."""
    if temperature >= 66:
        return 255
    if temperature <= 19:
        return 0
    blue = 138.5177312231 * math.log(temperature - 10) - 305.0447927307
    return _clamp(blue)

def color_temperature_mired_to_kelvin(mired_temperature: float) -> int:
    """Convert absolute mired shift to degrees kelvin."""
    return math.floor(1000000 / mired_temperature)

def color_temperature_kelvin_to_mired(kelvin_temperature: float) -> int:
    """Convert degrees kelvin to mired shift."""
    return math.floor(1000000 / kelvin_temperature)

def cross_product(p1: XYPoint, p2: XYPoint) -> float:
    """Calculate the cross product of two XYPoints."""
    return float(p1.x * p2.y - p1.y * p2.x)

def get_distance_between_two_points(one: XYPoint, two: XYPoint) -> float:
    """Calculate the distance between two XYPoints."""
    dx = one.x - two.x
    dy = one.y - two.y
    return math.sqrt(dx * dx + dy * dy)

def get_closest_point_to_line(A: XYPoint, B: XYPoint, P: XYPoint) -> XYPoint:
    """Find the closest point from P to a line defined by A and B.

    This point will be reproducible by the lamp
    as it is on the edge of the gamut.
    """
    AP = XYPoint(P.x - A.x, P.y - A.y)
    AB = XYPoint(B.x - A.x, B.y - A.y)
    ab2 = AB.x * AB.x + AB.y * AB.y
    ap_ab = AP.x * AB.x + AP.y * AB.y
    t = ap_ab / ab2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return XYPoint(A.x + AB.x * t, A.y + AB.y * t)

def get_closest_point_to_point(xy_tuple: Tuple[float, float], Gamut: GamutType) -> Tuple[float, float]:
    """Get the closest matching color within the gamut of the light.

    Should only be used if the supplied color is outside of the color gamut.
    """
    xy_point = XYPoint(xy_tuple[0], xy_tuple[1])
    pAB = get_closest_point_to_line(Gamut.red, Gamut.green, xy_point)
    pAC = get_closest_point_to_line(Gamut.blue, Gamut.red, xy_point)
    pBC = get_closest_point_to_line(Gamut.green, Gamut.blue, xy_point)
    dAB = get_distance_between_two_points(xy_point, pAB)
    dAC = get_distance_between_two_points(xy_point, pAC)
    dBC = get_distance_between_two_points(xy_point, pBC)
    lowest = dAB
    closest_point = pAB
    if dAC < lowest:
        lowest = dAC
        closest_point = pAC
    if dBC < lowest:
        lowest = dBC
        closest_point = pBC
    cx = closest_point.x
    cy = closest_point.y
    return (cx, cy)

def check_point_in_lamps_reach(p: Tuple[float, float], Gamut: GamutType) -> bool:
    """Check if the provided XYPoint can be recreated by a Hue lamp."""
    v1 = XYPoint(Gamut.green.x - Gamut.red.x, Gamut.green.y - Gamut.red.y)
    v2 = XYPoint(Gamut.blue.x - Gamut.red.x, Gamut.blue.y - Gamut.red.y)
    q = XYPoint(p[0] - Gamut.red.x, p[1] - Gamut.red.y)
    s = cross_product(q, v2) / cross_product(v1, v2)
    t = cross_product(v1, q) / cross_product(v1, v2)
    return s >= 0.0 and t >= 0.0 and (s + t <= 1.0)

def check_valid_gamut(Gamut: GamutType) -> bool:
    """Check if the supplied gamut is valid."""
    v1 = XYPoint(Gamut.green.x - Gamut.red.x, Gamut.green.y - Gamut.red.y)
    v2 = XYPoint(Gamut.blue.x - Gamut.red.x, Gamut.blue.y - Gamut.red.y)
    not_on_line = cross_product(v1, v2) > 0.0001
    red_valid = Gamut.red.x >= 0 and Gamut.red.x <= 1 and (Gamut.red.y >= 0) and (Gamut.red.y <= 1)
    green_valid = Gamut.green.x >= 0 and Gamut.green.x <= 1 and (Gamut.green.y >= 0) and (Gamut.green.y <= 1)
    blue_valid = Gamut.blue.x >= 0 and Gamut.blue.x <= 1 and (Gamut.blue.y >= 0) and (Gamut.blue.y <= 1)
    return not_on_line and red_valid and green_valid and blue_valid

def brightness_to_value(low_high_range: Tuple[int, int], brightness: int) -> float:
    """Given a brightness_scale convert a brightness to a single value.

    Do not include 0 if the light is off for value 0.

    Given a brightness low_high_range of (1,100) this function
    will return:

    255: 100.0
    127: ~49.8039
    10: ~3.9216
    """
    return scale_to_ranged_value((1, 255), low_high_range, brightness)

def value_to_brightness(low_high_range: Tuple[int, int], value: int) -> int:
    """Given a brightness_scale convert a single value to a brightness.

    Do not include 0 if the light is off for value 0.

    Given a brightness low_high_range of (1,100) this function
    will return:

    100: 255
    50: 128
    4: 10

    The value will be clamped between 1..255 to ensure valid value.
    """
    return min(255, max(1, round(scale_to_ranged_value(low_high_range, (1, 255), value))))
