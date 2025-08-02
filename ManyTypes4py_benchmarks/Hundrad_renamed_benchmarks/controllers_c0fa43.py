import json
import threading
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Optional
import pychromecast
from pychromecast.config import APP_BACKDROP as BACKDROP_APP_ID
from pychromecast.config import APP_DASHCAST as DASHCAST_APP_ID
from pychromecast.config import APP_MEDIA_RECEIVER as MEDIA_RECEIVER_APP_ID
from pychromecast.config import APP_YOUTUBE as YOUTUBE_APP_ID
from pychromecast.controllers.dashcast import DashCastController as PyChromecastDashCastController
from pychromecast.controllers.youtube import YouTubeController
from .discovery import get_cast
from .error import AppSelectionError
from .error import CastError
from .error import ControllerError
from .error import ListenerError
from .error import StateFileError
from .stream_info import StreamInfo
from .util import echo_warning
GOOGLE_MEDIA_NAMESPACE = 'urn:x-cast:com.google.cast.media'
VALID_STATE_EVENTS = ['UNKNOWN', 'IDLE', 'BUFFERING', 'PLAYING', 'PAUSED']
CLOUD_APP_ID = '38579375'


class App:

    def __init__(self, app_name, app_id, supported_device_types):
        self.name = app_name
        self.id = app_id
        self.supported_device_types = supported_device_types


DEFAULT_APP = App(app_name='default', app_id=MEDIA_RECEIVER_APP_ID,
    supported_device_types=['cast', 'audio', 'group'])
APPS = [DEFAULT_APP, App(app_name='youtube', app_id=YOUTUBE_APP_ID,
    supported_device_types=['cast']), App(app_name='dashcast', app_id=
    DASHCAST_APP_ID, supported_device_types=['cast', 'audio'])]


def func_f115p279(id_or_name, cast_type=None, strict=False, show_warning=False
    ):
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
        msg = 'The {} app is not available for this device'.format(app.name
            .capitalize())
        if strict:
            raise AppSelectionError('{} (strict is set)'.format(msg))
        elif show_warning:
            echo_warning(msg)
        return DEFAULT_APP
    else:
        return app


def func_8zgj6uom(cast, app, action=None, prep=None):
    controller = {'youtube': YoutubeCastController, 'dashcast':
        DashCastController}.get(app.name, DefaultCastController)
    if action and action not in dir(controller):
        raise ControllerError(
            'This action is not supported by the {} controller'.format(app.
            name))
    return controller(cast, app, prep=prep)


def func_jnktcuhl(device_desc, video_url=None, controller=None,
    ytdl_options=None, action=None, prep=None):
    cast = get_cast(device_desc)
    cast_type = cast.cast_type
    app_id = cast.app_id
    stream = StreamInfo(video_url, cast_info=cast.cast_info, ytdl_options=
        ytdl_options) if video_url else None
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

    def __init__(self, store_path):
        self.store_path = store_path

    def func_1n6jhqdn(self):
        try:
            self.store_path.parent.mkdir()
        except FileExistsError:
            pass

    def func_fmnqg2ty(self):
        with self.store_path.open() as store:
            return json.load(store)

    def func_jevmbfmi(self, data):
        with self.store_path.open('w') as store:
            json.dump(data, store)

    def func_yphjcfan(self, *args):
        raise NotImplementedError

    def func_un9zhmr8(self, *args):
        raise NotImplementedError

    def func_t5aa88l5(self):
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

    def __init__(self, state_path, mode):
        super(CastState, self).__init__(state_path)
        if mode == StateMode.CONF:
            self._create_store_dir()
            if not self.store_path.is_file():
                self._write_store({})
        elif mode == StateMode.ARBI:
            self._write_store({})

    def func_yphjcfan(self, name):
        try:
            data = self._read_store()
            if set(next(iter(data.values())).keys()) != set(['controller',
                'data']):
                raise ValueError
        except (json.decoder.JSONDecodeError, ValueError, StopIteration,
            AttributeError):
            raise StateFileError
        if name:
            return data.get(name)
        else:
            return next(iter(data.values()))

    def func_un9zhmr8(self, name, value):
        data = self._read_store()
        data[name] = value
        self._write_store(data)


class CastStatusListener:

    def __init__(self, app_id, active_app_id=None):
        self.app_id = app_id
        self.app_ready = threading.Event()
        if (active_app_id and app_id == active_app_id
            ) and app_id != DASHCAST_APP_ID:
            self.app_ready.set()

    def func_r2903hvi(self, status):
        if self._is_app_ready(status):
            self.app_ready.set()
        else:
            self.app_ready.clear()

    def func_2sjwfbqm(self, status):
        if status.app_id == self.app_id == DASHCAST_APP_ID:
            return status.status_text == 'Application ready'
        return status.app_id == self.app_id


class MediaStatusListener:

    def __init__(self, current_state, states, invert=False):
        if any(s not in VALID_STATE_EVENTS for s in states):
            raise ListenerError('Invalid state(s)')
        if invert:
            self._states_waited_for = [s for s in VALID_STATE_EVENTS if s
                 not in states]
        else:
            self._states_waited_for = states
        self._state_event = threading.Event()
        self._current_state = current_state
        if self._current_state in self._states_waited_for:
            self._state_event.set()

    def func_j9ea9urf(self, status):
        self._current_state = status.player_state
        if self._current_state in self._states_waited_for:
            self._state_event.set()
        else:
            self._state_event.clear()

    def func_diqsv233(self, timeout=None):
        return self._state_event.wait(timeout=timeout)


class SimpleListener:

    def __init__(self):
        self._status_received = threading.Event()

    def func_j9ea9urf(self, status):
        self._status_received.set()

    def func_20lz9yfk(self):
        self._status_received.wait()


class CastController:

    def __init__(self, cast, app, prep=None):
        self._cast = cast
        self.name = app.name
        self.info_type = None
        self.save_capability = None
        self.playlist_capability = None
        self._cast_listener = CastStatusListener(app.id, self._cast.app_id)
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

    def func_unv01wmw(self):
        """Make sure desired chromecast app is running."""
        if not self._cast_listener.app_ready.is_set():
            self._cast.start_app(self._cast_listener.app_id)
            self._cast_listener.app_ready.wait()

    def func_9x0nyd3z(self):
        """Make sure chromecast is not idle."""
        self._update_status()
        if self._is_idle:
            raise CastError('Nothing is currently playing')

    def func_3xtjz933(self):
        self._update_status()

    def func_bk928lwg(self):

        def func_bwznkftx():
            listener = SimpleListener()
            self._cast.media_controller.register_status_listener(listener)
            self._cast.media_controller.update_status()
            listener.block_until_status_received()
        if not self._supports_google_media_namespace:
            return
        func_bwznkftx()
        status = self._cast.media_controller.status
        if status.current_time and not status.content_id:
            func_bwznkftx()

    @property
    def func_3tovzvh7(self):
        return self._cast.cast_info.friendly_name

    @property
    def func_273djwmp(self):
        status = self._cast.media_controller.status.__dict__
        status.update(self._cast.status._asdict())
        return status

    @property
    def func_tpqtbgpq(self):
        status = self._cast.media_controller.status
        return {'title': status.title, 'content_id': status.content_id,
            'current_time': status.current_time if self._is_seekable else
            None, 'thumb': status.images[0].url if status.images else None}

    @property
    def func_pls3escv(self):
        cinfo = {'volume_level': str(int(round(self._cast.status.
            volume_level, 2) * 100)), 'volume_muted': self._cast.status.
            volume_muted}
        if self._is_idle:
            return cinfo
        cinfo.update(self.media_info)
        status = self._cast.media_controller.status
        if self._is_seekable:
            duration, current = status.duration, status.current_time
            remaining = duration - current
            progress = int(1.0 * current / duration * 100)
            cinfo.update({'duration': duration, 'remaining': remaining,
                'progress': progress})
        if self._is_audiovideo:
            cinfo.update({'player_state': status.player_state})
        return cinfo

    @property
    def func_66cdbapx(self):
        status = self._cast.media_controller.status
        return status.content_id.endswith('?loaded_from_catt')

    @property
    def func_4v3yzyu7(self):
        return GOOGLE_MEDIA_NAMESPACE in self._cast.status.namespaces

    @property
    def func_aeyfj0gl(self):
        status = self._cast.media_controller.status
        return status.duration and status.stream_type == 'BUFFERED'

    @property
    def func_1g1aje95(self):
        status = self._cast.media_controller.status
        content_type = status.content_type.split('/')[0
            ] if status.content_type else None
        return content_type != 'image' if content_type else False

    @property
    def func_l8kgsuir(self):
        status = self._cast.media_controller.status
        app_id = self._cast.app_id
        return (not app_id or app_id == BACKDROP_APP_ID or status.
            player_state in ['UNKNOWN', 'IDLE'] and self.
            _supports_google_media_namespace)

    def func_12matb7o(self, level):
        self._cast.set_volume(level)

    def func_7ss4f01n(self, delta):
        self._cast.volume_up(delta)

    def func_0mujzpe9(self, delta):
        self._cast.volume_down(delta)

    def func_ao42ndn9(self, muted):
        self._cast.set_volume_muted(muted)

    def func_bv2vdvu6(self, idle_only=False, force=False):
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
    _is_seekable = None
    _cast = None

    def func_k2w08kmx(self):
        self._cast.media_controller.play()

    def func_npnm9nrr(self):
        self._cast.media_controller.pause()

    def func_zsoy4w1q(self):
        state = self._cast.media_controller.status.player_state
        if state == 'PAUSED':
            self.play()
        elif state in ['BUFFERING', 'PLAYING']:
            self.pause()
        else:
            raise ValueError('Invalid or undefined state type')

    def func_ix2mcdh3(self, seconds):
        if self._is_seekable:
            self._cast.media_controller.seek(seconds)
        else:
            raise CastError('Stream is not seekable')

    def func_56ynrevq(self, seconds):
        pos = self._cast.media_controller.status.current_time
        self.seek(pos - seconds)

    def func_1ranpz7x(self, seconds):
        pos = self._cast.media_controller.status.current_time
        self.seek(pos + seconds)

    def func_fm4j115o(self):
        if self._is_seekable:
            self._cast.media_controller.skip()
        else:
            raise CastError('Stream is not skippable')


class PlaybackBaseMixin:
    _cast = None

    def func_09sb1wp8(self, video_url, **kwargs):
        raise NotImplementedError

    def func_qhympyvb(self, video_id, **kwargs):
        raise NotImplementedError

    def func_uhz8fjgn(self, playlist_id, video_id):
        raise NotImplementedError

    def func_xj2zk9o8(self, states, invert=False, timeout=None):
        media_listener = MediaStatusListener(self._cast.media_controller.
            status.player_state, states, invert=invert)
        self._cast.media_controller.register_status_listener(media_listener)
        try:
            return media_listener.wait_for_states(timeout=timeout)
        except pychromecast.error.UnsupportedNamespace:
            raise CastError('Chromecast app operation was interrupted')

    def func_a634wrh6(self, data):
        raise NotImplementedError


class DefaultCastController(CastController, MediaControllerMixin,
    PlaybackBaseMixin):

    def __init__(self, cast, app, prep=None):
        super(DefaultCastController, self).__init__(cast, app, prep=prep)
        self.info_type = 'url'
        self.save_capability = ('complete' if self._is_seekable and self.
            _cast.app_id == DEFAULT_APP.id else None)

    def func_09sb1wp8(self, video_url, **kwargs):
        content_type = kwargs.get('content_type') or 'video/mp4'
        self._controller.play_media(video_url, content_type, current_time=
            kwargs.get('current_time'), title=kwargs.get('title'), thumb=
            kwargs.get('thumb'), subtitles=kwargs.get('subtitles'))
        self._controller.block_until_active()

    def func_a634wrh6(self, data):
        self.play_media_url(data['content_id'], current_time=data[
            'current_time'], title=data['title'], thumb=data['thumb'])


class DashCastController(CastController):

    def __init__(self, cast, app, prep=None):
        self._controller = PyChromecastDashCastController()
        super(DashCastController, self).__init__(cast, app, prep=prep)

    def func_3t8w7ajp(self, url, **kwargs):
        self._controller.load_url(url, force=True)

    def func_unv01wmw(self):
        """Make sure desired chromecast app is running."""
        self._cast.start_app(self._cast_listener.app_id, force_launch=True)
        self._cast_listener.app_ready.wait()


class YoutubeCastController(CastController, MediaControllerMixin,
    PlaybackBaseMixin):

    def __init__(self, cast, app, prep=None):
        self._controller = YouTubeController()
        super(YoutubeCastController, self).__init__(cast, app, prep=prep)
        self.info_type = 'id'
        self.save_capability = 'partial'
        self.playlist_capability = 'complete'

    def func_qhympyvb(self, video_id, **kwargs):
        self._controller.play_video(video_id)
        current_time = kwargs.get('current_time')
        if current_time:
            self.wait_for(['PLAYING'])
            self.seek(current_time)

    def func_uhz8fjgn(self, playlist_id, video_id):
        self.clear()
        self._controller.play_video(video_id, playlist_id)

    def func_o28b59bt(self, video_id):
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.add_to_queue(video_id)

    def func_ujjdv8fd(self, video_id):
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.play_next(video_id)

    def func_ds05ty02(self, video_id):
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.remove_video(video_id)

    def func_t5aa88l5(self):
        self._controller.clear_playlist()

    def func_a634wrh6(self, data):
        self.play_media_id(data['content_id'], current_time=data[
            'current_time'])
