from __future__ import annotations
import colorsys
import math
from typing import NamedTuple
import attr

class RGBColor(NamedTuple):
    """RGB hex values."""
    COLOR: attr.ib()

class XYPoint(NamedTuple):
    """Represents a CIE 1931 XY coordinate pair."""
    x: float
    y: float

class GamutType(NamedTuple):
    """Represents the Gamut of a light."""
    red: XYPoint
    green: XYPoint
    blue: XYPoint

def color_name_to_rgb(color_name: str) -> RGBColor:
    """Convert color name to RGB hex value."""
    hex_value = COLORS.get(color_name.replace(' ', '').lower())
    if not hex_value:
        raise ValueError('Unknown color')
    return hex_value

def color_RGB_to_xy(iR: int, iG: int, iB: int, Gamut: GamutType | None = None) -> tuple[float, float]:
    """Convert from RGB color to XY color."""
    return color_RGB_to_xy_brightness(iR, iG, iB, Gamut)[:2]

def color_RGB_to_xy_brightness(iR: int, iG: int, iB: int, Gamut: GamutType | None = None) -> tuple[float, float, int]:
    """Convert from RGB color to XY color."""
    # ... (rest of the function remains the same)

def color_xy_to_RGB(vX: float, vY: float, Gamut: GamutType | None = None) -> tuple[int, int, int]:
    """Convert from XY to a normalized RGB."""
    # ... (rest of the function remains the same)

def color_xy_brightness_to_RGB(vX: float, vY: float, ibrightness: int, Gamut: GamutType | None = None) -> tuple[int, int, int]:
    """Convert from XYZ to RGB."""
    # ... (rest of the function remains the same)

def color_hsv_to_RGB(fH: float, fS: float, fB: float) -> tuple[int, int, int]:
    """Convert a hsb into its rgb representation."""
    # ... (rest of the function remains the same)

def color_RGB_to_hsv(iR: int, iG: int, iB: int) -> tuple[float, float, float]:
    """Convert an rgb color to its hsv representation."""
    # ... (rest of the function remains the same)

def color_xy_to_hs(vX: float, vY: float, Gamut: GamutType | None = None) -> tuple[float, float]:
    """Convert an xy color to its hs representation."""
    # ... (rest of the function remains the same)

def color_hs_to_xy(iH: float, iS: float, Gamut: GamutType | None = None) -> tuple[float, float]:
    """Convert an hs color to its xy representation."""
    # ... (rest of the function remains the same)

def match_max_scale(input_colors: tuple[int, int, int], output_colors: tuple[int, int, int]) -> tuple[int, int, int]:
    """Match the maximum value of the output to the input."""
    # ... (rest of the function remains the same)

def color_rgb_to_rgbw(r: int, g: int, b: int) -> tuple[int, int, int, int]:
    """Convert an rgb color to an rgbw representation."""
    # ... (rest of the function remains the same)

def color_rgbw_to_rgb(r: int, g: int, b: int, w: int) -> tuple[int, int, int]:
    """Convert an rgbw color to an rgb representation."""
    # ... (rest of the function remains the same)

def color_rgb_to_rgbww(r: int, g: int, b: int, min_kelvin: int, max_kelvin: int) -> tuple[int, int, int, int, int]:
    """Convert an rgb color to an rgbww representation."""
    # ... (rest of the function remains the same)

def color_rgbww_to