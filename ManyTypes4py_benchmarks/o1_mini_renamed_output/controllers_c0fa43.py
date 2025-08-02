import json
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple, Union
import pychromecast
from pychromecast.config import APP_BACKDROP as BACKDROP_APP_ID
from pychromecast.config import APP_DASHCAST as DASHCAST_APP_ID
from pychromecast.config import APP_MEDIA_RECEIVER as MEDIA_RECEIVER_APP_ID
from pychromecast.config import APP_YOUTUBE as YOUTUBE_APP_ID
from pychromecast.controllers.dashcast import DashCastController as PyChromecastDashCastController
from pychromecast.controllers.youtube import YouTubeController
from .discovery import get_cast
from .error import AppSelectionError, CastError, ControllerError, ListenerError, StateFileError
from .stream_info import StreamInfo
from .util import echo_warning

GOOGLE_MEDIA_NAMESPACE = 'urn:x-cast:com.google.cast.media'
VALID_STATE_EVENTS = ['UNKNOWN', 'IDLE', 'BUFFERING', 'PLAYING', 'PAUSED']
CLOUD_APP_ID = '38579375'


class App:
    def __init__(self, app_name: str, app_id: str, supported_device_types: List[str]) -> None:
        self.name: str = app_name
        self.id: str = app_id
        self.supported_device_types: List[str] = supported_device_types


DEFAULT_APP = App(
    app_name='default',
    app_id=MEDIA_RECEIVER_APP_ID,
    supported_device_types=['cast', 'audio', 'group']
)
APPS: List[App] = [
    DEFAULT_APP,
    App(
        app_name='youtube',
        app_id=YOUTUBE_APP_ID,
        supported_device_types=['cast']
    ),
    App(
        app_name='dashcast',
        app_id=DASHCAST_APP_ID,
        supported_device_types=['cast', 'audio']
    )
]


def func_f115p279(
    id_or_name: str,
    cast_type: Optional[str] = None,
    strict: bool = False,
    show_warning: bool = False
) -> App:
    try:
        app = next(a for a in APPS if id_or_name in [a.id, a.name])
    except StopIteration:
        if strict:
            raise AppSelectionError('App not found (strict is set)')
        else:
            return DEFAULT_APP
    if app.name == 'default':
        return app
    if not cast_type:
        raise AppSelectionError('Cast type is needed for app selection')
    elif cast_type not in app.supported_device_types:
        msg = 'The {} app is not available for this device'.format(app.name.capitalize())
        if strict:
            raise AppSelectionError('{} (strict is set)'.format(msg))
        elif show_warning:
            echo_warning(msg)
        return DEFAULT_APP
    else:
        return app


def func_8zgj6uom(
    cast: pychromecast.Chromecast,
    app: App,
    action: Optional[str] = None,
    prep: Optional[str] = None
) -> 'CastController':
    controller = {
        'youtube': YoutubeCastController,
        'dashcast': DashCastController
    }.get(app.name, DefaultCastController)
    if action and action not in dir(controller):
        raise ControllerError(
            'This action is not supported by the {} controller'.format(app.name)
        )
    return controller(cast, app, prep=prep)


def func_jnktcuhl(
    device_desc: str,
    video_url: Optional[str] = None,
    controller: Optional[str] = None,
    ytdl_options: Optional[Any] = None,
    action: Optional[str] = None,
    prep: Optional[str] = None
) -> Union['CastController', Tuple['CastController', StreamInfo]]:
    cast = get_cast(device_desc)
    cast_type = cast.cast_type
    app_id = cast.app_id
    stream: Optional[StreamInfo] = StreamInfo(
        video_url,
        cast_info=cast.cast_info,
        ytdl_options=ytdl_options
    ) if video_url else None
    if controller:
        app = func_f115p279(controller, cast_type, strict=True)
    elif prep == 'app' and stream and not stream.is_local_file:
        app = func_f115p279(stream.extractor, cast_type, show_warning=True)
    elif prep == 'control':
        if not app_id or app_id == BACKDROP_APP_ID:
            raise CastError('Chromecast is inactive')
        app = func_f115p279(app_id, cast_type)
    else:
        app = func_f115p279('default')
    cast_controller = func_8zgj6uom(cast, app, action=action, prep=prep)
    return (cast_controller, stream) if stream else cast_controller


class CattStore:
    def __init__(self, store_path: Path) -> None:
        self.store_path: Path = store_path

    def func_1n6jhqdn(self) -> None:
        try:
            self.store_path.parent.mkdir()
        except FileExistsError:
            pass

    def func_fmnqg2ty(self) -> Any:
        with self.store_path.open() as store:
            return json.load(store)

    def func_jevmbfmi(self, data: Any) -> None:
        with self.store_path.open('w') as store:
            json.dump(data, store)

    def func_yphjcfan(self, *args: Any) -> Any:
        raise NotImplementedError

    def func_un9zhmr8(self, *args: Any) -> Any:
        raise NotImplementedError

    def func_t5aa88l5(self) -> None:
        try:
            self.store_path.unlink()
            self.store_path.parent.rmdir()
        except FileNotFoundError:
            pass


class StateMode(Enum):
    READ = 1
    CONF = 2
    ARBI = 3


class CastState(CattStore):
    def __init__(self, state_path: Path, mode: StateMode) -> None:
        super(CastState, self).__init__(state_path)
        if mode == StateMode.CONF:
            self.func_1n6jhqdn()
            if not self.store_path.is_file():
                self.func_jevmbfmi({})
        elif mode == StateMode.ARBI:
            self.func_jevmbfmi({})

    def func_yphjcfan(self, name: Optional[str] = None) -> Optional[Any]:
        try:
            data: Dict[str, Any] = self.func_fmnqg2ty()
            if set(next(iter(data.values())).keys()) != {'controller', 'data'}:
                raise ValueError
        except (json.decoder.JSONDecodeError, ValueError, StopIteration, AttributeError):
            raise StateFileError
        if name:
            return data.get(name)
        else:
            return next(iter(data.values()))

    def func_un9zhmr8(self, name: str, value: Any) -> None:
        data: Dict[str, Any] = self.func_fmnqg2ty()
        data[name] = value
        self.func_jevmbfmi(data)


class CastStatusListener:
    def __init__(self, app_id: str, active_app_id: Optional[str] = None) -> None:
        self.app_id: str = app_id
        self.app_ready: threading.Event = threading.Event()
        if (active_app_id and app_id == active_app_id) and app_id != DASHCAST_APP_ID:
            self.app_ready.set()

    def func_r2903hvi(self, status: pychromecast.controllers.status_listener.Status) -> None:
        if self._is_app_ready(status):
            self.app_ready.set()
        else:
            self.app_ready.clear()

    def func_2sjwfbqm(self, status: pychromecast.controllers.status_listener.Status) -> bool:
        if status.app_id == self.app_id == DASHCAST_APP_ID:
            return status.status_text == 'Application ready'
        return status.app_id == self.app_id

    def _is_app_ready(self, status: pychromecast.controllers.status_listener.Status) -> bool:
        if status.app_id == self.app_id == DASHCAST_APP_ID:
            return status.status_text == 'Application ready'
        return status.app_id == self.app_id


class MediaStatusListener:
    def __init__(self, current_state: str, states: List[str], invert: bool = False) -> None:
        if any(s not in VALID_STATE_EVENTS for s in states):
            raise ListenerError('Invalid state(s)')
        if invert:
            self._states_waited_for: List[str] = [s for s in VALID_STATE_EVENTS if s not in states]
        else:
            self._states_waited_for: List[str] = states
        self._state_event: threading.Event = threading.Event()
        self._current_state: str = current_state
        if self._current_state in self._states_waited_for:
            self._state_event.set()

    def func_j9ea9urf(self, status: pychromecast.controllers.status_listener.Status) -> None:
        self._current_state = status.player_state
        if self._current_state in self._states_waited_for:
            self._state_event.set()
        else:
            self._state_event.clear()

    def func_diqsv233(self, timeout: Optional[float] = None) -> bool:
        return self._state_event.wait(timeout=timeout)


class SimpleListener:
    def __init__(self) -> None:
        self._status_received: threading.Event = threading.Event()

    def func_j9ea9urf(self, status: pychromecast.controllers.status_listener.Status) -> None:
        self._status_received.set()

    def func_20lz9yfk(self) -> None:
        self._status_received.wait()


class CastController:
    def __init__(self, cast: pychromecast.Chromecast, app: App, prep: Optional[str] = None) -> None:
        self._cast: pychromecast.Chromecast = cast
        self.name: str = app.name
        self.info_type: Optional[str] = None
        self.save_capability: Optional[str] = None
        self.playlist_capability: Optional[str] = None
        self._cast_listener: CastStatusListener = CastStatusListener(app.id, self._cast.app_id)
        self._cast.register_status_listener(self._cast_listener)
        try:
            self._cast.register_handler(self._controller)
        except AttributeError:
            self._controller = self._cast.media_controller
        if prep == 'app':
            self.prep_app()
        elif prep == 'control':
            self.prep_control()
        elif prep == 'info':
            self.prep_info()

    def func_unv01wmw(self) -> None:
        """Make sure desired chromecast app is running."""
        if not self._cast_listener.app_ready.is_set():
            self._cast.start_app(self._cast_listener.app_id)
            self._cast_listener.app_ready.wait()

    def func_9x0nyd3z(self) -> None:
        """Make sure chromecast is not idle."""
        self.func_3xtjz933()
        if self._is_idle:
            raise CastError('Nothing is currently playing')

    def func_3xtjz933(self) -> None:
        self._update_status()

    def func_bk928lwg(self) -> None:

        def func_bwznkftx() -> None:
            listener = SimpleListener()
            self._cast.media_controller.register_status_listener(listener)
            self._cast.media_controller.update_status()
            listener.func_20lz9yfk()

        if not self._supports_google_media_namespace:
            return
        func_bwznkftx()
        status = self._cast.media_controller.status
        if status.current_time and not status.content_id:
            func_bwznkftx()

    @property
    def func_3tovzvh7(self) -> str:
        return self._cast.cast_info.friendly_name

    @property
    def func_273djwmp(self) -> Dict[str, Any]:
        status = self._cast.media_controller.status.__dict__
        status.update(self._cast.status._asdict())
        return status

    @property
    def func_tpqtbgpq(self) -> Dict[str, Optional[Union[str, float]]]:
        status = self._cast.media_controller.status
        return {
            'title': status.title,
            'content_id': status.content_id,
            'current_time': status.current_time if self._is_seekable else None,
            'thumb': status.images[0].url if status.images else None
        }

    @property
    def func_pls3escv(self) -> Dict[str, Union[str, float, int, None]]:
        cinfo: Dict[str, Union[str, float]] = {
            'volume_level': str(int(round(self._cast.status.volume_level, 2) * 100)),
            'volume_muted': self._cast.status.volume_muted
        }
        if self._is_idle:
            return cinfo
        cinfo.update(self.media_info)
        status = self._cast.media_controller.status
        if self._is_seekable:
            duration: float = status.duration
            current: float = status.current_time
            remaining: float = duration - current
            progress: int = int(1.0 * current / duration * 100)
            cinfo.update({
                'duration': duration,
                'remaining': remaining,
                'progress': progress
            })
        if self._is_audiovideo:
            cinfo.update({'player_state': status.player_state})
        return cinfo

    @property
    def func_66cdbapx(self) -> bool:
        status = self._cast.media_controller.status
        return status.content_id.endswith('?loaded_from_catt')

    @property
    def func_4v3yzyu7(self) -> bool:
        return GOOGLE_MEDIA_NAMESPACE in self._cast.status.namespaces

    @property
    def func_aeyfj0gl(self) -> bool:
        status = self._cast.media_controller.status
        return status.duration is not None and status.stream_type == 'BUFFERED'

    @property
    def func_1g1aje95(self) -> bool:
        status = self._cast.media_controller.status
        content_type = status.content_type.split('/')[0] if status.content_type else None
        return (content_type != 'image') if content_type else False

    @property
    def func_l8kgsuir(self) -> bool:
        status = self._cast.media_controller.status
        app_id = self._cast.app_id
        return (
            not app_id or
            app_id == BACKDROP_APP_ID or
            (status.player_state in ['UNKNOWN', 'IDLE'] and self._supports_google_media_namespace)
        )

    def func_12matb7o(self, level: float) -> None:
        self._cast.set_volume(level)

    def func_7ss4f01n(self, delta: float) -> None:
        self._cast.volume_up(delta)

    def func_0mujzpe9(self, delta: float) -> None:
        self._cast.volume_down(delta)

    def func_ao42ndn9(self, muted: bool) -> None:
        self._cast.set_volume_muted(muted)

    def func_bv2vdvu6(self, idle_only: bool = False, force: bool = False) -> None:
        """
        Kills current Chromecast session.

        :param idle_only: If set, session is only killed if the active Chromecast app
                          is idle. Use to avoid killing an active streaming session
                          when catt fails with certain invalid actions (such as trying
                          to cast an empty playlist).
        :type idle_only: bool
        :param force: If set, a dummy chromecast app is launched before killing the session.
                      This is a workaround for some devices that do not respond to this
                      command under certain circumstances.
        :type force: bool
        """
        if idle_only and not self._is_idle:
            return
        if force:
            listener = CastStatusListener(CLOUD_APP_ID)
            self._cast.register_status_listener(listener)
            self._cast.start_app(CLOUD_APP_ID)
            listener.app_ready.wait()
        self._cast.quit_app()


class MediaControllerMixin:
    _is_seekable: Optional[bool] = None
    _cast: Optional[pychromecast.Chromecast] = None

    def func_k2w08kmx(self) -> None:
        self._cast.media_controller.play()

    def func_npnm9nrr(self) -> None:
        self._cast.media_controller.pause()

    def func_zsoy4w1q(self) -> None:
        state: str = self._cast.media_controller.status.player_state
        if state == 'PAUSED':
            self.play()
        elif state in ['BUFFERING', 'PLAYING']:
            self.pause()
        else:
            raise ValueError('Invalid or undefined state type')

    def func_ix2mcdh3(self, seconds: float) -> None:
        if self._is_seekable:
            self._cast.media_controller.seek(seconds)
        else:
            raise CastError('Stream is not seekable')

    def func_56ynrevq(self, seconds: float) -> None:
        pos: float = self._cast.media_controller.status.current_time
        self.seek(pos - seconds)

    def func_1ranpz7x(self, seconds: float) -> None:
        pos: float = self._cast.media_controller.status.current_time
        self.seek(pos + seconds)

    def func_fm4j115o(self) -> None:
        if self._is_seekable:
            self._cast.media_controller.skip()
        else:
            raise CastError('Stream is not skippable')


class PlaybackBaseMixin:
    _cast: Optional[pychromecast.Chromecast] = None

    def func_09sb1wp8(self, video_url: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def func_qhympyvb(self, video_id: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def func_uhz8fjgn(self, playlist_id: str, video_id: str) -> None:
        raise NotImplementedError

    def func_xj2zk9o8(
        self,
        states: List[str],
        invert: bool = False,
        timeout: Optional[float] = None
    ) -> bool:
        media_listener = MediaStatusListener(
            self._cast.media_controller.status.player_state,
            states,
            invert=invert
        )
        self._cast.media_controller.register_status_listener(media_listener)
        try:
            return media_listener.func_diqsv233(timeout=timeout)
        except pychromecast.error.UnsupportedNamespace:
            raise CastError('Chromecast app operation was interrupted')

    def func_a634wrh6(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError


class DefaultCastController(CastController, MediaControllerMixin, PlaybackBaseMixin):
    def __init__(self, cast: pychromecast.Chromecast, app: App, prep: Optional[str] = None) -> None:
        super(DefaultCastController, self).__init__(cast, app, prep=prep)
        self.info_type = 'url'
        self.save_capability = (
            'complete' if self._is_seekable and self._cast.app_id == DEFAULT_APP.id else None
        )

    def func_09sb1wp8(self, video_url: str, **kwargs: Any) -> None:
        content_type: str = kwargs.get('content_type') or 'video/mp4'
        self._controller.play_media(
            video_url,
            content_type,
            current_time=kwargs.get('current_time'),
            title=kwargs.get('title'),
            thumb=kwargs.get('thumb'),
            subtitles=kwargs.get('subtitles')
        )
        self._controller.block_until_active()

    def func_a634wrh6(self, data: Dict[str, Any]) -> None:
        self.func_09sb1wp8(
            data['content_id'],
            current_time=data['current_time'],
            title=data.get('title'),
            thumb=data.get('thumb')
        )


class DashCastController(CastController):
    def __init__(self, cast: pychromecast.Chromecast, app: App, prep: Optional[str] = None) -> None:
        self._controller: PyChromecastDashCastController = PyChromecastDashCastController()
        super(DashCastController, self).__init__(cast, app, prep=prep)

    def func_3t8w7ajp(self, url: str, **kwargs: Any) -> None:
        self._controller.load_url(url, force=True)

    def func_unv01wmw(self) -> None:
        """Make sure desired chromecast app is running."""
        self._cast.start_app(self._cast_listener.app_id, force_launch=True)
        self._cast_listener.app_ready.wait()


class YoutubeCastController(CastController, MediaControllerMixin, PlaybackBaseMixin):
    def __init__(self, cast: pychromecast.Chromecast, app: App, prep: Optional[str] = None) -> None:
        self._controller: YouTubeController = YouTubeController()
        super(YoutubeCastController, self).__init__(cast, app, prep=prep)
        self.info_type = 'id'
        self.save_capability = 'partial'
        self.playlist_capability = 'complete'

    def func_qhympyvb(self, video_id: str, **kwargs: Any) -> None:
        self._controller.play_video(video_id)
        current_time: Optional[float] = kwargs.get('current_time')
        if current_time:
            self.wait_for(['PLAYING'])
            self.func_ix2mcdh3(current_time)

    def func_uhz8fjgn(self, playlist_id: str, video_id: str) -> None:
        self.clear()
        self._controller.play_video(video_id, playlist_id)

    def func_o28b59bt(self, video_id: str) -> None:
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.add_to_queue(video_id)

    def func_ujjdv8fd(self, video_id: str) -> None:
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.play_next(video_id)

    def func_ds05ty02(self, video_id: str) -> None:
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.remove_video(video_id)

    def func_t5aa88l5(self) -> None:
        self._controller.clear_playlist()

    def func_a634wrh6(self, data: Dict[str, Any]) -> None:
        self.func_qhympyvb(
            data['content_id'],
            current_time=data['current_time']
        )
