"""Color util methods."""

from __future__ import annotations

import colorsys
import math
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast

import attr

from .scaling import scale_to_ranged_value


class RGBColor(NamedTuple):
    """RGB hex values."""

    r: int
    g: int
    b: int


# Official CSS3 colors from w3.org:
# https://www.w3.org/TR/2010/PR-css3-color-20101028/#html4
# names do not have spaces in them so that we can compare against
# requests more easily (by removing spaces from the requests as well).
# This lets "dark seagreen" and "dark sea green" both match the same
# color "darkseagreen".
COLORS: Dict[str, RGBColor] = {
    "aliceblue": RGBColor(240, 248, 255),
    "antiquewhite": RGBColor(250, 235, 215),
    "aqua": RGBColor(0, 255, 255),
    "aquamarine": RGBColor(127, 255, 212),
    "azure": RGBColor(240, 255, 255),
    "beige": RGBColor(245, 245, 220),
    "bisque": RGBColor(255, 228, 196),
    "black": RGBColor(0, 0, 0),
    "blanchedalmond": RGBColor(255, 235, 205),
    "blue": RGBColor(0, 0, 255),
    "blueviolet": RGBColor(138, 43, 226),
    "brown": RGBColor(165, 42, 42),
    "burlywood": RGBColor(222, 184, 135),
    "cadetblue": RGBColor(95, 158, 160),
    "chartreuse": RGBColor(127, 255, 0),
    "chocolate": RGBColor(210, 105, 30),
    "coral": RGBColor(255, 127, 80),
    "cornflowerblue": RGBColor(100, 149, 237),
    "cornsilk": RGBColor(255, 248, 220),
    "crimson": RGBColor(220, 20, 60),
    "cyan": RGBColor(0, 255, 255),
    "darkblue": RGBColor(0, 0, 139),
    "darkcyan": RGBColor(0, 139, 139),
    "darkgoldenrod": RGBColor(184, 134, 11),
    "darkgray": RGBColor(169, 169, 169),
    "darkgreen": RGBColor(0, 100, 0),
    "darkgrey": RGBColor(169, 169, 169),
    "darkkhaki": RGBColor(189, 183, 107),
    "darkmagenta": RGBColor(139, 0, 139),
    "darkolivegreen": RGBColor(85, 107, 47),
    "darkorange": RGBColor(255, 140, 0),
    "darkorchid": RGBColor(153, 50, 204),
    "darkred": RGBColor(139, 0, 0),
    "darksalmon": RGBColor(233, 150, 122),
    "darkseagreen": RGBColor(143, 188, 143),
    "darkslateblue": RGBColor(72, 61, 139),
    "darkslategray": RGBColor(47, 79, 79),
    "darkslategrey": RGBColor(47, 79, 79),
    "darkturquoise": RGBColor(0, 206, 209),
    "darkviolet": RGBColor(148, 0, 211),
    "deeppink": RGBColor(255, 20, 147),
    "deepskyblue": RGBColor(0, 191, 255),
    "dimgray": RGBColor(105, 105, 105),
    "dimgrey": RGBColor(105, 105, 105),
    "dodgerblue": RGBColor(30, 144, 255),
    "firebrick": RGBColor(178, 34, 34),
    "floralwhite": RGBColor(255, 250, 240),
    "forestgreen": RGBColor(34, 139, 34),
    "fuchsia": RGBColor(255, 0, 255),
    "gainsboro": RGBColor(220, 220, 220),
    "ghostwhite": RGBColor(248, 248, 255),
    "gold": RGBColor(255, 215, 0),
    "goldenrod": RGBColor(218, 165, 32),
    "gray": RGBColor(128, 128, 128),
    "green": RGBColor(0, 128, 0),
    "greenyellow": RGBColor(173, 255, 47),
    "grey": RGBColor(128, 128, 128),
    "honeydew": RGBColor(240, 255, 240),
    "hotpink": RGBColor(255, 105, 180),
    "indianred": RGBColor(205, 92, 92),
    "indigo": RGBColor(75, 0, 130),
    "ivory": RGBColor(255, 255, 240),
    "khaki": RGBColor(240, 230, 140),
    "lavender": RGBColor(230, 230, 250),
    "lavenderblush": RGBColor(255, 240, 245),
    "lawngreen": RGBColor(124, 252, 0),
    "lemonchiffon": RGBColor(255, 250, 205),
    "lightblue": RGBColor(173, 216, 230),
    "lightcoral": RGBColor(240, 128, 128),
    "lightcyan": RGBColor(224, 255, 255),
    "lightgoldenrodyellow": RGBColor(250, 250, 210),
    "lightgray": RGBColor(211, 211, 211),
    "lightgreen": RGBColor(144, 238, 144),
    "lightgrey": RGBColor(211, 211, 211),
    "lightpink": RGBColor(255, 182, 193),
    "lightsalmon": RGBColor(255, 160, 122),
    "lightseagreen": RGBColor(32, 178, 170),
    "lightskyblue": RGBColor(135, 206, 250),
    "lightslategray": RGBColor(119, 136, 153),
    "lightslategrey": RGBColor(119, 136, 153),
    "lightsteelblue": RGBColor(176, 196, 222),
    "lightyellow": RGBColor(255, 255, 224),
    "lime": RGBColor(0, 255, 0),
    "limegreen": RGBColor(50, 205, 50),
    "linen": RGBColor(250, 240, 230),
    "magenta": RGBColor(255, 0, 255),
    "maroon": RGBColor(128, 0, 0),
    "mediumaquamarine": RGBColor(102, 205, 170),
    "mediumblue": RGBColor(0, 0, 205),
    "mediumorchid": RGBColor(186, 85, 211),
    "mediumpurple": RGBColor(147, 112, 219),
    "mediumseagreen": RGBColor(60, 179, 113),
    "mediumslateblue": RGBColor(123, 104, 238),
    "mediumspringgreen": RGBColor(0, 250, 154),
    "mediumturquoise": RGBColor(72, 209, 204),
    "mediumvioletred": RGBColor(199, 21, 133),
    "midnightblue": RGBColor(25, 25, 112),
    "mintcream": RGBColor(245, 255, 250),
    "mistyrose": RGBColor(255, 228, 225),
    "moccasin": RGBColor(255, 228, 181),
    "navajowhite": RGBColor(255, 222, 173),
    "navy": RGBColor(0, 0, 128),
    "navyblue": RGBColor(0, 0, 128),
    "oldlace": RGBColor(253, 245, 230),
    "olive": RGBColor(128, 128, 0),
    "olivedrab": RGBColor(107, 142, 35),
    "orange": RGBColor(255, 165, 0),
    "orangered": RGBColor(255, 69, 0),
    "orchid": RGBColor(218, 112, 214),
    "palegoldenrod": RGBColor(238, 232, 170),
    "palegreen": RGBColor(152, 251, 152),
    "paleturquoise": RGBColor(175, 238, 238),
    "palevioletred": RGBColor(219, 112, 147),
    "papayawhip": RGBColor(255, 239, 213),
    "peachpuff": RGBColor(255, 218, 185),
    "peru": RGBColor(205, 133, 63),
    "pink": RGBColor(255, 192, 203),
    "plum": RGBColor(221, 160, 221),
    "powderblue": RGBColor(176, 224, 230),
    "purple": RGBColor(128, 0, 128),
    "red": RGBColor(255, 0, 0),
    "rosybrown": RGBColor(188, 143, 143),
    "royalblue": RGBColor(65, 105, 225),
    "saddlebrown": RGBColor(139, 69, 19),
    "salmon": RGBColor(250, 128, 114),
    "sandybrown": RGBColor(244, 164, 96),
    "seagreen": RGBColor(46, 139, 87),
    "seashell": RGBColor(255, 245, 238),
    "sienna": RGBColor(160, 82, 45),
    "silver": RGBColor(192, 192, 192),
    "skyblue": RGBColor(135, 206, 235),
    "slateblue": RGBColor(106, 90, 205),
    "slategray": RGBColor(112, 128, 144),
    "slategrey": RGBColor(112, 128, 144),
    "snow": RGBColor(255, 250, 250),
    "springgreen": RGBColor(0, 255, 127),
    "steelblue": RGBColor(70, 130, 180),
    "tan": RGBColor(210, 180, 140),
    "teal": RGBColor(0, 128, 128),
    "thistle": RGBColor(216, 191, 216),
    "tomato": RGBColor(255, 99, 71),
    "turquoise": RGBColor(64, 224, 208),
    "violet": RGBColor(238, 130, 238),
    "wheat": RGBColor(245, 222, 179),
    "white": RGBColor(255, 255, 255),
    "whitesmoke": RGBColor(245, 245, 245),
    "yellow": RGBColor(255, 255, 0),
    "yellowgreen": RGBColor(154, 205, 50),
    # And...
    "homeassistant": RGBColor(24, 188, 242),
}


@attr.s()
class XYPoint:
    """Represents a CIE 1931 XY coordinate pair."""

    x: float = attr.ib()
    y: float = attr.ib()


@attr.s()
class GamutType:
    """Represents the Gamut of a light."""

    # ColorGamut = gamut(xypoint(xR,yR),xypoint(xG,yG),xypoint(xB,yB))
    red: XYPoint = attr.ib()
    green: XYPoint = attr.ib()
    blue: XYPoint = attr.ib()


def color_name_to_rgb(color_name: str) -> RGBColor:
    """Convert color name to RGB hex value."""
    # COLORS map has no spaces in it, so make the color_name have no
    # spaces in it as well for matching purposes
    hex_value = COLORS.get(color_name.replace(" ", "").lower())
    if not hex_value:
        raise ValueError("Unknown color")

    return hex_value


def color_RGB_to_xy(
    iR: int, iG: int, iB: int, Gamut: Optional[GamutType] = None
) -> Tuple[float, float]:
    """Convert from RGB color to XY color."""
    return color_RGB_to_xy_brightness(iR, iG, iB, Gamut)[:2]


# Taken from:
# https://github.com/PhilipsHue/PhilipsHueSDK-iOS-OSX/blob/00187a3/ApplicationDesignNotes/RGB%20to%20xy%20Color%20conversion.md
# License: Code is given as is. Use at your own risk and discretion.
def color_RGB_to_xy_brightness(
    iR: int, iG: int, iB: int, Gamut: Optional[GamutType] = None
) -> Tuple[float, float, int]:
    """Convert from RGB color to XY color."""
    if iR + iG + iB == 0:
        return 0.0, 0.0, 0

    R = iR / 255
    B = iB / 255
    G = iG / 255

    # Gamma correction
    R = pow((R + 0.055) / (1.0 + 0.055), 2.4) if (R > 0.04045) else (R / 12.92)
    G = pow((G + 0.055) / (1.0 + 0.055), 2.4) if (G > 0.04045) else (G / 12.92)
    B = pow((B + 0.055) / (1.0 + 0.055), 2.4) if (B > 0.04045) else (B / 12.92)

    # Wide RGB D65 conversion formula
    X = R * 0.664511 + G * 0.154324 + B * 0.162028
    Y = R * 0.283881 + G * 0.668433 + B * 0.047685
    Z = R * 0.000088 + G * 0.072310 + B * 0.986039

    # Convert XYZ to xy
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    # Brightness
    Y = min(Y, 1)
    brightness = round(Y * 255)

    # Check if the given xy value is within the color-reach of the lamp.
    if Gamut:
        in_reach = check_point_in_lamps_reach((x, y), Gamut)
        if not in_reach:
            xy_closest = get_closest_point_to_point((x, y), Gamut)
            x = xy_closest[0]
            y = xy_closest[1]

    return round(x, 3), round(y, 3), brightness


def color_xy_to_RGB(
    vX: float, vY: float, Gamut: Optional[GamutType] = None
) -> Tuple[int, int, int]:
    """Convert from XY to a normalized RGB."""
    return color_xy_brightness_to_RGB(vX, vY, 255, Gamut)


# Converted to Python from Obj-C, original source from:
# https://github.com/PhilipsHue/PhilipsHueSDK-iOS-OSX/blob/00187a3/ApplicationDesignNotes/RGB%20to%20xy%20Color%20conversion.md
def color_xy_brightness_to_RGB(
    vX: float, vY: float, ibrightness: int, Gamut: Optional[GamutType] = None
) -> Tuple[int, int, int]:
    """Convert from XYZ to RGB."""
    if Gamut and not check_point_in_lamps_reach((vX, vY), Gamut):
        xy_closest = get_closest_point_to_point((vX, vY), Gamut)
        vX = xy_closest[0]
        vY = xy_closest[1]

    brightness = ibrightness / 255.0
    if brightness == 0.0:
        return (0, 0, 0)

    Y = brightness

    if vY == 0.0:
        vY += 0.00000000001

    X = (Y / vY) * vX
    Z = (Y / vY) * (1 - vX - vY)

    # Convert to RGB using Wide RGB D65 conversion.
    r = X * 1.656492 - Y * 0.354851 - Z * 0.255038
    g = -X * 0.707196 + Y * 1.655397 + Z * 0.036152
    b = X * 0.051713 - Y * 0.121364 + Z * 1.011530

    # Apply reverse gamma correction.
    r, g, b = (
        12.92 * x if (x <= 0.0031308) else ((1.0 + 0.055) * pow(x, (1.0 / 2.4)) - 0.055)
        for x in (r, g, b)
    )

    #