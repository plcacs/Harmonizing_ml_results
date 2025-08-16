from typing import List, Optional
from pychromecast import Chromecast
from .controllers import CastController, get_app, get_controller
from .discovery import get_cast_with_ip, get_cast_with_name, get_casts
from .error import APIError, CastError
from .stream_info import StreamInfo

class CattDevice:

    def __init__(self, name: str = '', ip_addr: str = '', lazy: bool = False) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def _create_cast(self) -> None:
        ...

    def _create_controller(self) -> None:
        ...

    @property
    def controller(self) -> CastController:
        ...

    def play_url(self, url: str, resolve: bool = False, block: bool = False, subtitle_url: Optional[str] = None, **kwargs) -> None:
        ...

    def stop(self) -> None:
        ...

    def play(self) -> None:
        ...

    def pause(self) -> None:
        ...

    def seek(self, seconds: int) -> None:
        ...

    def rewind(self, seconds: int) -> None:
        ...

    def ffwd(self, seconds: int) -> None:
        ...

    def volume(self, level: float) -> None:
        ...

    def volumeup(self, delta: float) -> None:
        ...

    def volumedown(self, delta: float) -> None:
        ...

    def volumemute(self, muted: bool) -> None:
        ...

def discover() -> List[CattDevice]:
    ...
