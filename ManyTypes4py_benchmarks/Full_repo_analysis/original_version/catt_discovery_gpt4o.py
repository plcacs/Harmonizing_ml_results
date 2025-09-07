from typing import List, Optional, Union

import pychromecast

from .error import CastError
from .util import is_ipaddress

DEFAULT_PORT: int = 8009


def get_casts(names: Optional[List[str]] = None) -> List[pychromecast.Chromecast]:
    if names:
        cast_infos, browser = pychromecast.discovery.discover_listed_chromecasts(friendly_names=names)
    else:
        cast_infos, browser = pychromecast.discovery.discover_chromecasts()

    casts: List[pychromecast.Chromecast] = [pychromecast.get_chromecast_from_cast_info(c, browser.zc) for c in cast_infos]

    for cast in casts:
        cast.wait()

    browser.stop_discovery()
    casts.sort(key=lambda c: c.cast_info.friendly_name)
    return casts


def get_cast_infos() -> List[pychromecast.CastInfo]:
    return [c.cast_info for c in get_casts()]


def get_cast_with_name(cast_name: Optional[str]) -> Optional[pychromecast.Chromecast]:
    casts: List[pychromecast.Chromecast] = get_casts([cast_name]) if cast_name else get_casts()
    return casts[0] if casts else None


def get_cast_with_ip(cast_ip: str, port: int = DEFAULT_PORT) -> Optional[pychromecast.Chromecast]:
    device_info = pychromecast.discovery.get_device_info(cast_ip)
    if not device_info:
        return None

    host: tuple = (cast_ip, DEFAULT_PORT, device_info.uuid, device_info.model_name, device_info.friendly_name)
    cast: pychromecast.Chromecast = pychromecast.get_chromecast_from_host(host)
    cast.wait()
    return cast


def cast_ip_exists(cast_ip: str) -> bool:
    return bool(get_cast_with_ip(cast_ip))


def get_cast(cast_desc: Optional[str] = None) -> pychromecast.Chromecast:
    cast: Optional[pychromecast.Chromecast] = None

    if cast_desc and is_ipaddress(cast_desc):
        cast = get_cast_with_ip(cast_desc)
        if not cast:
            msg: str = "No device found at {}".format(cast_desc)
            raise CastError(msg)
    else:
        cast = get_cast_with_name(cast_desc)
        if not cast:
            msg: str = 'Specified device "{}" not found'.format(cast_desc) if cast_desc else "No devices found"
            raise CastError(msg)

    return cast
