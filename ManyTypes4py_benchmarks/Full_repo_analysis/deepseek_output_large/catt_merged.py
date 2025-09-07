import sys
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set, NamedTuple, cast
if sys.version_info.major < 3:
    print('This program requires Python 3 and above to run.')
    sys.exit(1)
__codename__: str = 'Zaniest Zapper'

import sys
if sys.version_info.major < 3:
    print('This program requires Python 3 and above to run.')
    sys.exit(1)
__codename__: str = 'Zaniest Zapper'

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

    def __init__(self, name: str = '', ip_addr: str = '', lazy: bool = False) -> None:
        """
        Class to easily interface with a ChromeCast.

        :param name: Name of ChromeCast device to interface with.
                     Either name of ip-address must be supplied.
        :param ip_addr: Ip-address of device to interface with.
                       Either name of ip-address must be supplied.
        :param lazy: Postpone first connection attempt to device
                     until first playback action is attempted.
        """
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

    def play_url(self, url: str, resolve: bool = False, block: bool = False, subtitle_url: Optional[str] = None, **kwargs: Any) -> None:
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

    def stop(self) -> None:
        """Stop playback."""
        self.controller.kill()

    def play(self) -> None:
        """Resume playback of paused content."""
        self.controller.prep_control()
        self.controller.play()

    def pause(self) -> None:
        """Pause playback of content."""
        self.controller.prep_control()
        self.controller.pause()

    def seek(self, seconds: int) -> None:
        """
        Seek to arbitrary position in content.

        :param seconds: Position in seconds.
        """
        self.controller.prep_control()
        self.controller.seek(seconds)

    def rewind(self, seconds: int) -> None:
        """
        Seek backwards in content by arbitrary amount of seconds.

        :param seconds: Seek amount in seconds.
        """
        self.controller.prep_control()
        self.controller.rewind(seconds)

    def ffwd(self, seconds: int) -> None:
        """
        Seek forward in content by arbitrary amount of seconds.

        :param seconds: Seek amount in seconds.
        """
        self.controller.prep_control()
        self.controller.ffwd(seconds)

    def volume(self, level: float) -> None:
        """
        Set volume to arbitrary level.

        :param level: Volume level (valid range: 0.0-1.0).
        """
        self.controller.volume(level)

    def volumeup(self, delta: float) -> None:
        """
        Raise volume by arbitrary delta.

        :param delta: Volume delta (valid range: 0.0-1.0).
        """
        self.controller.volumeup(delta)

    def volumedown(self, delta: float) -> None:
        """
        Lower volume by arbitrary delta.

        :param delta: Volume delta (valid range: 0.0-1.0).
        """
        self.controller.volumedown(delta)

    def volumemute(self, muted: bool) -> None:
        """
        Enable mute on supported devices.

        :param muted: Whether to mute the device. (valid values: true or false).
        """
        self.controller.volumemute(muted)

def discover() -> List[CattDevice]:
    """Perform discovery of devices present on local network, and return result."""
    return [CattDevice(ip_addr=c.socket_client.host) for c in get_casts()]

import configparser
import random
import sys
import time
try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import click
from . import __codename__
from .controllers import CastState
from .controllers import setup_cast
from .controllers import StateFileError
from .controllers import StateMode
from .discovery import cast_ip_exists
from .discovery import get_cast_infos
from .error import CastError
from .error import CattUserError
from .error import CliError
from .http_server import serve_file
from .subs_info import SubsInfo
from .util import echo_json
from .util import echo_status
from .util import echo_warning
from .util import hunt_subtitles
from .util import is_ipaddress
CONFIG_DIR: Path = Path(click.get_app_dir('catt'))
CONFIG_PATH: Path = Path(CONFIG_DIR, 'catt.cfg')
STATE_PATH: Path = Path(CONFIG_DIR, 'state.json')
WAIT_PLAY_TIMEOUT: int = 30
PROGRAM_NAME: str = 'catt'
try:
    VERSION: str = version(PROGRAM_NAME)
except Exception:
    VERSION: str = '0.0.0u'

class CattTimeParamType(click.ParamType):

    def convert(self, value: str, param: Optional[click.Parameter], ctx: click.Context) -> int:
        try:
            tdesc: List[int] = [int(x) for x in value.split(':')]
            tlen: int = len(tdesc)
            if tlen > 1 and any((t > 59 for t in tdesc)) or tlen > 3:
                raise ValueError
        except ValueError:
            self.fail('{} is not a valid time description.'.format(value))
        tdesc.reverse()
        return sum((tdesc[p] * 60 ** p for p in range(tlen)))
CATT_TIME: CattTimeParamType = CattTimeParamType()

class YtdlOptParamType(click.ParamType):

    def convert(self, value: str, param: Optional[click.Parameter], ctx: click.Context) -> Tuple[str, Any]:
        if '=' not in value:
            self.fail('{} is not a valid key/value pair.'.format(value))
        (ykey, yval) = value.split('=', 1)
        yval = {'true': True, 'false': False}.get(yval.lower().strip(), yval)
        return (ykey, yval)
YTDL_OPT: YtdlOptParamType = YtdlOptParamType()

def process_url(ctx: click.Context, param: click.Parameter, value: str) -> str:
    if value == '-':
        stdin_text = click.get_text_stream('stdin')
        if not stdin_text.isatty():
            value = stdin_text.read().strip()
        else:
            raise CliError('No input received from stdin')
    if '://' not in value:
        if ctx.info_name != 'cast':
            raise CliError('Local file not allowed as argument to this command')
        if not Path(value).is_file():
            raise CliError('The chosen file does not exist')
    return value

def process_path(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[Path]:
    path: Optional[Path] = Path(value) if value else None
    if path and (path.is_dir() or not path.parent.exists()):
        raise CliError('The specified path is invalid')
    return path

def process_subtitles(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    pval: str = urlparse(value).path if '://' in value else value
    if not pval.lower().endswith(('.srt', '.vtt')):
        raise CliError('Invalid subtitles format, only srt and vtt are supported')
    if '://' not in value and (not Path(value).is_file()):
        raise CliError('Subtitles file [{}] does not exist'.format(value))
    return value

def process_device(device_desc: str, aliases: Dict[str, str]) -> str:
    """
    Resolve real device name when value is an alias.

    :param device_desc: Can be an ip-address or a name (alias or real name).
    :type device_desc: str
    :param aliases: Dictionary of device aliases and their corresponding real names.
    :type aliases: Dict[str, str]
    """
    if is_ipaddress(device_desc):
        return device_desc
    elif device_desc:
        return aliases.get(device_desc.lower(), device_desc)
    else:
        return device_desc

def fail_if_no_ip(ipaddr: Optional[str]) -> None:
    if not ipaddr:
        raise CliError('Local IP-address could not be determined')

def create_server_thread(filename: str, address: str, port: int, content_type: str, single_req: bool = False) -> Thread:
    thr: Thread = Thread(target=serve_file, args=(filename, address, port, content_type, single_req))
    thr.setDaemon(True)
    thr.start()
    return thr
CONTEXT_SETTINGS: Dict[str, Any] = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('-d', '--device', metavar='NAME_OR_IP', help='Select Chromecast device.')
@click.version_option(version=VERSION, prog_name=PROGRAM_NAME, message='%(prog)s v%(version)s, ' + __codename__ + '.')
@click.pass_context
def cli(ctx: click.Context, device: str) -> None:
    device_from_config: Optional[str] = ctx.obj['options'].get('device')
    ctx.obj['selected_device'] = process_device(device or device_from_config, ctx.obj['aliases'])
    ctx.obj['selected_device_is_from_cli'] = bool(device)

@cli.command(short_help='Send a video to a Chromecast for playing.')
@click.argument('video_url', callback=process_url)
@click.option('-s', '--subtitles', callback=process_subtitles, metavar='SUB', help='Specify a subtitles file.')
@click.option('-f', '--force-default', is_flag=True, help="Force use of the default Chromecast app (use if a custom app doesn't work).")
@click.option('-r', '--random-play', is_flag=True, help='Play random item from playlist, if applicable.')
@click.option('--no-subs', is_flag=True, default=False, help="Don't try to load subtitles automatically from the local folder.")
@click.option('-n', '--no-playlist', is_flag=True, help='Play only video, if url contains both video and playlist ids.')
@click.option('-y', '--ytdl-option', type=YTDL_OPT, multiple=True, metavar='OPT', help='yt-dlp option. Should be passed as `-y option=value`, and can be specified multiple times (implies --force-default).')
@click.option('-t', '--seek-to', type=CATT_TIME, metavar='TIME', help='Start playback at specific timestamp.')
@click.option('-b', '--block', is_flag=True, help='Keep catt process alive until playback has ended. Only useful when casting remote files, as catt is already running a server when casting local files. Currently exits after playback of single media, so not useful with playlists yet.')
@click.pass_obj
def cast(settings: Dict[str, Any], video_url: str, subtitles: Optional[str], force_default: bool, random_play: bool, no_subs: bool, no_playlist: bool, ytdl_option: Tuple[Tuple[str, Any], ...], seek_to: Optional[int], block: bool = False) -> None:
    controller: Optional[str] = 'default' if force_default or ytdl_option else None
    playlist_playback: bool = False
    st_thr: Optional[Thread] = None
    su_thr: Optional[Thread] = None
    subs: Optional[SubsInfo] = None
    (cst, stream) = setup_cast(settings['selected_device'], video_url=video_url, prep='app', controller=controller, ytdl_options=ytdl_option)
    media_is_image: bool = stream.guessed_content_category == 'image'
    local_or_remote: str = 'local' if stream.is_local_file else 'remote'
    if stream.is_local_file:
        fail_if_no_ip(stream.local_ip)
        st_thr = create_server_thread(video_url, stream.local_ip, stream.port, stream.guessed_content_type, single_req=media_is_image)
    elif stream.is_playlist and (not (no_playlist and stream.video_id)):
        if stream.playlist_length == 0:
            cst.kill(idle_only=True)
            raise CliError('Playlist is empty')
        if not random_play and cst.playlist_capability and stream.playlist_all_ids:
            playlist_playback = True
        else:
            if random_play:
                entry: int = random.randrange(0, stream.playlist_length)
            else:
                echo_warning('Playlist playback not possible, playing first video')
                entry = 0
            stream.set_playlist_entry(entry)
    if playlist_playback:
        click.echo('Casting remote playlist {}...'.format(video_url))
        video_id: str = stream.video_id or stream.playlist_all_ids[0]
        cst.play_playlist(stream.playlist_id, video_id=video_id)
    else:
        if not subtitles and (not no_subs) and stream.is_local_file:
            subtitles = hunt_subtitles(video_url)
        if subtitles:
            fail_if_no_ip(stream.local_ip)
            subs = SubsInfo(subtitles, stream.local_ip, stream.port + 1)
            su_thr = create_server_thread(subs.file, subs.local_ip, subs.port, 'text/vtt;charset=utf-8', single_req=True)
        click.echo('Casting {} file {}...'.format(local_or_remote, video_url))
        click.echo('{} "{}" on "{}"...'.format('Showing' if media_is_image else 'Playing', stream.video_title, cst.cc_name))
        if cst.info_type == 'url':
            cst.play_media_url(stream.video_url, title=stream.video_title, content_type=stream.guessed_content_type, subtitles=subs.url if subs else None, thumb=stream.video_thumbnail, current_time=seek_to)
        elif cst.info_type == 'id':
            cst.play_media_id(stream.video_id, current_time=seek_to)
        else:
            raise ValueError('Invalid or undefined info type')
    if stream.is_local_file or subs:
        click.echo('Serving local file(s).')
    if not media_is_image and (stream.is_local_file or block):
        if not cst.wait_for(['PLAYING'], timeout=WAIT_PLAY_TIMEOUT):
            raise CliError('Playback of {} file has failed'.format(local_or_remote))
        cst.wait_for(['UNKNOWN', 'IDLE'])
    elif stream.is_local_file and media_is_image or subs:
        while st_thr and st_thr.is_alive() or (su_thr and su_thr.is_alive()):
            time.sleep(1)

@cli.command('cast_site', short_help='Cast any website to a Chromecast.')
@click.argument('url', callback=process_url)
@click.pass_obj
def cast_site(settings: Dict[str, Any], url: str) -> None:
    cst = setup_cast(settings['selected_device'], controller='dashcast', action='load_url', prep='app')
    click.echo('Casting {} on "{}"...'.format(url, cst.cc_name))
    cst.load_url(url)

@cli.command(short_help='Add a