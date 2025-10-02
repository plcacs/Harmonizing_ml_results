from typing import List, Optional, Union
from pychromecast import Chromecast
from .controllers import CastController
from .controllers import get_app
from .controllers import get_controller
from .discovery import get_cast_with_ip
from .discovery import get_cast_with_name
from .discovery import get_casts
from .error import APIError
from .error import CastError
from .stream_info import StreamInfo

class CattDevice:

    def __init__(self, name: str = '', ip_addr: str = '', lazy: bool = False) -> None:
        if not name and (not ip_addr):
            raise APIError('Neither name nor ip were supplied')
        self.name: str = name
        self.ip_addr: str = ip_addr
        self.uuid: Optional[str] = None
        self._cast: Optional[Chromecast] = None
        self._cast_controller: Optional[CastController] = None
        if not lazy:
            self._create_cast()

    def __repr__(self) -> str:
        return '<CattDevice: {}>'.format(self.name or self.ip_addr)

    def _create_cast(self) -> None:
        self._cast = get_cast_with_ip(self.ip_addr) if self.ip_addr else get_cast_with_name(self.name)
        if not self._cast:
            raise CastError('Device could not be found')
        self.name = self._cast.cast_info.friendly_name
        self.ip_addr = self._cast.cast_info.host
        self.uuid = self._cast.cast_info.uuid

    def _create_controller(self) -> None:
        self._cast_controller = get_controller(self._cast, get_app('default'))

    @property
    def controller(self) -> CastController:
        if not self._cast:
            self._create_cast()
        if not self._cast_controller:
            self._create_controller()
        return self._cast_controller

    def play_url(self, url: str, resolve: bool = False, block: bool = False, subtitle_url: Optional[str] = None, **kwargs) -> None:
        if resolve:
            stream = StreamInfo(url)
            url = stream.video_url
        self.controller.prep_app()
        self.controller.play_media_url(url, subtitles=subtitle_url, **kwargs)
        if self.controller.wait_for(['PLAYING'], timeout=10):
            if block:
                self.controller.wait_for(['UNKNOWN', 'IDLE'])
        else:
            raise APIError('Playback failed')

    def stop(self) -> None:
        self.controller.kill()

    def play(self) -> None:
        self.controller.prep_control()
        self.controller.play()

    def pause(self) -> None:
        self.controller.prep_control()
        self.controller.pause()

    def seek(self, seconds: Union[int, float]) -> None:
        self.controller.prep_control()
        self.controller.seek(seconds)

    def rewind(self, seconds: Union[int, float]) -> None:
        self.controller.prep_control()
        self.controller.rewind(seconds)

    def ffwd(self, seconds: Union[int, float]) -> None:
        self.controller.prep_control()
        self.controller.ffwd(seconds)

    def volume(self, level: float) -> None:
        self.controller.volume(level)

    def volumeup(self, delta: float) -> None:
        self.controller.volumeup(delta)

    def volumedown(self, delta: float) -> None:
        self.controller.volumedown(delta)

    def volumemute(self, muted: bool) -> None:
        self.controller.volumemute(muted)

def discover() -> List[CattDevice]:
    return [CattDevice(ip_addr=c.socket_client.host) for c in get_casts()]
