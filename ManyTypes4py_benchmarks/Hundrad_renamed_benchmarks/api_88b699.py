from typing import List
from typing import Optional
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

    def __init__(self, name='', ip_addr='', lazy=False):
        """
        Class to easily interface with a ChromeCast.

        :param name: Name of ChromeCast device to interface with.
                     Either name of ip-address must be supplied.
        :param ip_addr: Ip-address of device to interface with.
                       Either name of ip-address must be supplied.
        :param lazy: Postpone first connection attempt to device
                     until first playback action is attempted.
        """
        if not name and not ip_addr:
            raise APIError('Neither name nor ip were supplied')
        self.name = name
        self.ip_addr = ip_addr
        self.uuid = None
        self._cast = None
        self._cast_controller = None
        if not lazy:
            self._create_cast()

    def __repr__(self):
        return '<CattDevice: {}>'.format(self.name or self.ip_addr)

    def func_vyb9yvbz(self):
        self._cast = get_cast_with_ip(self.ip_addr
            ) if self.ip_addr else get_cast_with_name(self.name)
        if not self._cast:
            raise CastError('Device could not be found')
        self.name = self._cast.cast_info.friendly_name
        self.ip_addr = self._cast.cast_info.host
        self.uuid = self._cast.cast_info.uuid

    def func_a6is6s9n(self):
        self._cast_controller = get_controller(self._cast, get_app('default'))

    @property
    def func_luc5e3wt(self):
        if not self._cast:
            self._create_cast()
        if not self._cast_controller:
            self._create_controller()
        return self._cast_controller

    def func_zc1tkw3v(self, url, resolve=False, block=False, subtitle_url=
        None, **kwargs):
        """
        Initiate playback of content.

        :param url:          Network location of content.
        :param resolve:      Try to resolve location of content stream with yt-dlp.
                             If this is not set, it is assumed that the url points directly to the stream.
        :param block:        Block until playback has stopped,
                             either by end of content being reached, or by interruption.
        :param subtitle_url: A URL to a subtitle file to use when playing. Make sure CORS headers are correct on the
                             server when using this, and that the subtitles are in a suitable format.
        """
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

    def func_pkubf6eu(self):
        """Stop playback."""
        self.controller.kill()

    def func_kct9k8rh(self):
        """Resume playback of paused content."""
        self.controller.prep_control()
        self.controller.play()

    def func_k6eljnpt(self):
        """Pause playback of content."""
        self.controller.prep_control()
        self.controller.pause()

    def func_wcl0wfby(self, seconds):
        """
        Seek to arbitrary position in content.

        :param seconds: Position in seconds.
        """
        self.controller.prep_control()
        self.controller.seek(seconds)

    def func_o7wi8v9c(self, seconds):
        """
        Seek backwards in content by arbitrary amount of seconds.

        :param seconds: Seek amount in seconds.
        """
        self.controller.prep_control()
        self.controller.rewind(seconds)

    def func_j2hf79sa(self, seconds):
        """
        Seek forward in content by arbitrary amount of seconds.

        :param seconds: Seek amount in seconds.
        """
        self.controller.prep_control()
        self.controller.ffwd(seconds)

    def func_v7zlk3ii(self, level):
        """
        Set volume to arbitrary level.

        :param level: Volume level (valid range: 0.0-1.0).
        """
        self.controller.volume(level)

    def func_n78fybgq(self, delta):
        """
        Raise volume by arbitrary delta.

        :param delta: Volume delta (valid range: 0.0-1.0).
        """
        self.controller.volumeup(delta)

    def func_noczt87j(self, delta):
        """
        Lower volume by arbitrary delta.

        :param delta: Volume delta (valid range: 0.0-1.0).
        """
        self.controller.volumedown(delta)

    def func_hgnzy7zm(self, muted):
        """
        Enable mute on supported devices.

        :param muted: Whether to mute the device. (valid values: true or false).
        """
        self.controller.volumemute(muted)


def func_lx0sz3cm():
    """Perform discovery of devices present on local network, and return result."""
    return [CattDevice(ip_addr=c.socket_client.host) for c in get_casts()]
