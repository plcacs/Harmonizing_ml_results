"""Color util methods."""
from __future__ import annotations
import colorsys
import math
from typing import NamedTuple, Dict, Tuple, Optional, List, Union
import attr
from .scaling import scale_to_ranged_value

class RGBColor(NamedTuple):
    """RGB hex values."""
    red: int
    green: int
    blue: int

COLORS: Dict[str, RGBColor] = {
    'aliceblue': RGBColor(240, 248, 255),
    'antiquewhite': RGBColor(250, 235, 215),
    # ... (rest of the color definitions remain the same)
    'yellowgreen': RGBColor(154, 205, 50),
    'homeassistant': RGBColor(24, 188, 242)
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

def color_RGB_to_xy(iR: int, iG: int, iB: int, Gamut: Optional[GamutType] = None) -> Tuple[float, float]:
    """Convert from RGB color to XY color."""
    return color_RGB_to_xy_brightness(iR, iG, iB, Gamut)[:2]

def color_RGB_to_xy_brightness(iR: int, iG: int, iB: int, Gamut: Optional[GamutType] = None) -> Tuple[float, float, int]:
    """Convert from RGB color to XY color."""
    if iR + iG + iB == 0:
        return (0.0, 0.0, 0)
    R = iR / 255
    B = iB / 255
    G = iG / 255
    R = pow((R + 0.055) / (1.0 + 0.055), 2.4) if R > 0.04045 else R / 12.92
    G = pow((G + 0.055) / (1.0 + 0.055), 2.4) if G > 0.04045 else G / 12.92
    B = pow((B + 0.055) / (1.0 + 0.055), 2.4) if B > 0.04045 else B / 12.92
    X = R * 0.664511 + G * 0.154324 + B * 0.162028
    Y = R * 0.283881 + G * 0.668433 + B * 0.047685
    Z = R * 8.8e-05 + G * 0.07231 + B * 0.986039
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    Y = min(Y, 1)
    brightness = round(Y * 255)
    if Gamut:
        in_reach = check_point_in_lamps_reach((x, y), Gamut)
        if not in_reach:
            xy_closest = get_closest_point_to_point((x, y), Gamut)
            x = xy_closest[0]
            y = xy_closest[1]
    return (round(x, 3), round(y, 3), brightness)

def color_xy_to_RGB(vX: float, vY: float, Gamut: Optional[GamutType] = None) -> Tuple[int, int, int]:
    """Convert from XY to a normalized RGB."""
    return color_xy_brightness_to_RGB(vX, vY, 255, Gamut)

def color_xy_brightness_to_RGB(vX: float, vY: float, ibrightness: int, Gamut: Optional[GamutType] = None) -> Tuple[int, int, int]:
    """Convert from XYZ to RGB."""
    if Gamut and (not check_point_in_lamps_reach((vX, vY), Gamut)):
        xy_closest = get_closest_point_to_point((vX, vY), Gamut)
        vX = xy_closest[0]
        vY = xy_closest[1]
    brightness = ibrightness / 255.0
    if brightness == 0.0:
        return (0, 0, 0)
    Y = brightness
    if vY == 0.0:
        vY += 1e-11
    X = Y / vY * vX
    Z = Y / vY * (1 - vX - vY)
    r = X * 1.656492 - Y * 0.354851 - Z * 0.255038
    g = -X * 0.707196 + Y * 1.655397 + Z * 0.036152
    b = X * 0.051713 - Y * 0.121364 + Z * 1.01153
    r, g, b = (12.92 * x if x <= 0.0031308 else (1.0 + 0.055) * pow(x, 1.0 / 2.4) - 0.055 for x in (r, g, b))
    r, g, b = (max(0, x) for x in (r, g, b))
    max_component = max(r, g, b)
    if max_component > 1:
        r, g, b = (x / max_component for x in (r, g, b))
    ir, ig, ib = (int(x * 255) for x in (r, g, b))
    return (ir, ig, ib)

def color_hsb_to_RGB(fH: float, fS: float, fB: float) -> Tuple[int, int, int]:
    """Convert a hsb into its rgb representation."""
    if fS == 0.0:
        fV = int(fB * 255)
        return (fV, fV, fV)
    r = g = b = 0
    h = fH / 60
    f = h - float(math.floor(h))
    p = fB * (1 - fS)
    q = fB * (1 - fS * f)
    t = fB * (1 - fS * (1 - f))
    if int(h) == 0:
        r = int(fB * 255)
        g = int(t * 255)
        b = int(p * 255)
    elif int(h) == 1:
        r = int(q * 255)
        g = int(fB * 255)
        b = int(p * 255)
    elif int(h) == 2:
        r = int(p * 255)
        g = int(fB * 255)
        b = int(t * 255)
    elif int(h) == 3:
        r = int(p * 255)
        g = int(q * 255)
        b = int(fB * 255)
    elif int(h) == 4:
        r = int(t * 255)
        g = int(p * 255)
        b = int(fB * 255)
    elif int(h) == 5:
        r = int(fB * 255)
        g = int(p * 255)
        b = int(q * 255)
    return (r, g, b)

def color_RGB_to_hsv(iR: int, iG: int, iB: int) -> Tuple[float, float, float]:
    """Convert an rgb color to its hsv representation."""
    fHSV = colorsys.rgb_to_hsv(iR / 255.0, iG / 255.0, iB / 255.0)
    return (round(fHSV[0] * 360, 3), round(fHSV[1] * 100, 3), round(fHSV[2] * 100, 3))

def color_RGB_to_hs(iR: int, iG: int, iB: int) -> Tuple[float, float]:
    """Convert an rgb color to its hs representation."""
    return color_RGB_to_hsv(iR, iG, iB)[:2]

def color_hsv_to_RGB(iH: float, iS: float, iV: float) -> Tuple[int, int, int]:
    """Convert an hsv color into its rgb representation."""
    fRGB = colorsys.hsv_to_rgb(iH / 360, iS / 100, iV / 100)
    return (round(fRGB[0] * 255), round(fRGB[1] * 255), round(fRGB[2] * 255))

def color_hs_to_RGB(iH: float, iS: float) -> Tuple[int, int, int]:
    """Convert an hsv color into its rgb representation."""
    return color_hsv_to_RGB(iH, iS, 100)

def color_xy_to_hs(vX: float, vY: float, Gamut: Optional[GamutType] = None) -> Tuple[float, float]:
    """Convert an xy color to its hs representation."""
    h, s, _ = color_RGB_to_hsv(*color_xy_to_RGB(vX, vY, Gamut))
    return (h, s)

def color_hs_to_xy(iH: float, iS: float, Gamut: Optional[GamutType] = None) -> Tuple[float, float]:
    """Convert an hs color to its xy representation."""
    return color_RGB_to_xy(*color_hs_to_RGB(iH, iS), Gamut)

def match_max_scale(input_colors: Tuple[int, ...], output_colors: Tuple[Union[int, float], ...]) -> Tuple[int, ...]:
    """Match the maximum value of the output to the input."""
    max_in = max(input_colors)
    max_out = max(output_colors)
    if max_out == 0:
        factor = 0.0
    else:
        factor = max_in / max_out
    return tuple((int(round(i * factor)) for i in output_colors)

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

def rgb_hex_to_rgb_list(hex_string: str) -> List[int]:
    """Return an RGB color value list from a hex color string."""
    return [int(hex_string[i:i + len(hex_string) // 3], 16) for i in range(0, len(hex_string), len(hex_string) // 3)]

def color_temperature_to_hs(color_temperature_kelvin: int) -> Tuple[float, float]:
    """Return an hs color from a color temperature in Kelvin."""
    return color_RGB_to_hs(*color_temperature_to_rgb(color_temperature_kelvin))

def color_temperature_to_rgb(color_temperature_kelvin: int) -> Tuple[int, int, int]:
    """Return an RGB color from a color temperature in Kelvin."""
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
    """Convert color temperature in kelvin to rgbcw."""
    max_mireds = color_temperature_kelvin_to_mired(min_kelvin)
    min_mireds = color_temperature_kelvin_to_mired(max_kelvin)
    temperature = color_temperature_kelvin_to_mired(temperature)
    mired_range = max_mireds - min_mireds
    cold = (max_mireds - temperature) / mired_range * brightness
    warm = brightness - cold
    return (0, 0, 0, round(cold), round(warm))

def rgbww_to_color_temperature(rgbww: Tuple[int, int, int, int, int], min_kelvin: int, max_kelvin: int) -> Tuple[int, int]:
    """Convert rgbcw to color temperature in kelvin."""
    _, _, _, cold, warm = rgbww
    return _white_levels_to_color_temperature(cold, warm, min_kelvin, max_kelvin)

def _white_levels_to_color_temperature(cold: int, warm: int, min_kelvin: int, max_kelvin: int) -> Tuple[int, int]:
    """Convert whites to color temperature in kelvin."""
    max_mireds = color_temperature_kelvin_to_mired(min_kelvin)
    min_mireds = color_temperature_kelvin_to_mired(max_kelvin)
    brightness = warm / 255 + cold / 255
    if brightness == 0:
        return (min_kelvin, 0)
    return (round(color_temperature_mired_to_kelvin(cold / 255 / brightness * (min_mireds - max_mireds) + max_mireds)), min(255, round(brightness * 255)))

def color_xy_to_temperature(x: float, y: float) -> int:
    """Convert an xy color to a color temperature in Kelvin."""
    n = (x - 0.332) / (0.1858 - y)
    CCT = 437 * n ** 3 + 3601 * n ** 2 + 6861 * n + 5517
    return int(CCT)

def _clamp(color_component: float, minimum: int = 0, maximum: int = 255) -> int:
    """Clamp the given color component value between the given min and max values."""
    color_component_out = max(color_component, minimum)
    return min(color_component_out, maximum)

def _get_red(temperature: float) -> int:
    """Get the red component of the temperature in RGB space."""
    if temperature <= 66:
        return 255
    tmp_red = 329.698727446 * math.pow(temperature - 60, -0.1332047592)
    return _clamp(tmp