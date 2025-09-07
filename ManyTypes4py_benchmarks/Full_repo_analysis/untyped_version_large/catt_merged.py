# Merged Python files from catt
# Total files: 21


# File: catt\__init__.py

import sys
if sys.version_info.major < 3:
    print('This program requires Python 3 and above to run.')
    sys.exit(1)
__codename__ = 'Zaniest Zapper'

# File: catt\__init___gpt4o.py

import sys
if sys.version_info.major < 3:
    print('This program requires Python 3 and above to run.')
    sys.exit(1)
__codename__ = 'Zaniest Zapper'

# File: catt\api.py

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
        if not name and (not ip_addr):
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

    def _create_cast(self):
        self._cast = get_cast_with_ip(self.ip_addr) if self.ip_addr else get_cast_with_name(self.name)
        if not self._cast:
            raise CastError('Device could not be found')
        self.name = self._cast.cast_info.friendly_name
        self.ip_addr = self._cast.cast_info.host
        self.uuid = self._cast.cast_info.uuid

    def _create_controller(self):
        self._cast_controller = get_controller(self._cast, get_app('default'))

    @property
    def controller(self):
        if not self._cast:
            self._create_cast()
        if not self._cast_controller:
            self._create_controller()
        return self._cast_controller

    def play_url(self, url, resolve=False, block=False, subtitle_url=None, **kwargs):
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

    def stop(self):
        """Stop playback."""
        self.controller.kill()

    def play(self):
        """Resume playback of paused content."""
        self.controller.prep_control()
        self.controller.play()

    def pause(self):
        """Pause playback of content."""
        self.controller.prep_control()
        self.controller.pause()

    def seek(self, seconds):
        """
        Seek to arbitrary position in content.

        :param seconds: Position in seconds.
        """
        self.controller.prep_control()
        self.controller.seek(seconds)

    def rewind(self, seconds):
        """
        Seek backwards in content by arbitrary amount of seconds.

        :param seconds: Seek amount in seconds.
        """
        self.controller.prep_control()
        self.controller.rewind(seconds)

    def ffwd(self, seconds):
        """
        Seek forward in content by arbitrary amount of seconds.

        :param seconds: Seek amount in seconds.
        """
        self.controller.prep_control()
        self.controller.ffwd(seconds)

    def volume(self, level):
        """
        Set volume to arbitrary level.

        :param level: Volume level (valid range: 0.0-1.0).
        """
        self.controller.volume(level)

    def volumeup(self, delta):
        """
        Raise volume by arbitrary delta.

        :param delta: Volume delta (valid range: 0.0-1.0).
        """
        self.controller.volumeup(delta)

    def volumedown(self, delta):
        """
        Lower volume by arbitrary delta.

        :param delta: Volume delta (valid range: 0.0-1.0).
        """
        self.controller.volumedown(delta)

    def volumemute(self, muted):
        """
        Enable mute on supported devices.

        :param muted: Whether to mute the device. (valid values: true or false).
        """
        self.controller.volumemute(muted)

def discover():
    """Perform discovery of devices present on local network, and return result."""
    return [CattDevice(ip_addr=c.socket_client.host) for c in get_casts()]

# File: catt\cli.py

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
CONFIG_DIR = Path(click.get_app_dir('catt'))
CONFIG_PATH = Path(CONFIG_DIR, 'catt.cfg')
STATE_PATH = Path(CONFIG_DIR, 'state.json')
WAIT_PLAY_TIMEOUT = 30
PROGRAM_NAME = 'catt'
try:
    VERSION = version(PROGRAM_NAME)
except Exception:
    VERSION = '0.0.0u'

class CattTimeParamType(click.ParamType):

    def convert(self, value, param, ctx):
        try:
            tdesc = [int(x) for x in value.split(':')]
            tlen = len(tdesc)
            if tlen > 1 and any((t > 59 for t in tdesc)) or tlen > 3:
                raise ValueError
        except ValueError:
            self.fail('{} is not a valid time description.'.format(value))
        tdesc.reverse()
        return sum((tdesc[p] * 60 ** p for p in range(tlen)))
CATT_TIME = CattTimeParamType()

class YtdlOptParamType(click.ParamType):

    def convert(self, value, param, ctx):
        if '=' not in value:
            self.fail('{} is not a valid key/value pair.'.format(value))
        (ykey, yval) = value.split('=', 1)
        yval = {'true': True, 'false': False}.get(yval.lower().strip(), yval)
        return (ykey, yval)
YTDL_OPT = YtdlOptParamType()

def process_url(ctx, param, value):
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

def process_path(ctx, param, value):
    path = Path(value) if value else None
    if path and (path.is_dir() or not path.parent.exists()):
        raise CliError('The specified path is invalid')
    return path

def process_subtitles(ctx, param, value):
    if not value:
        return None
    pval = urlparse(value).path if '://' in value else value
    if not pval.lower().endswith(('.srt', '.vtt')):
        raise CliError('Invalid subtitles format, only srt and vtt are supported')
    if '://' not in value and (not Path(value).is_file()):
        raise CliError('Subtitles file [{}] does not exist'.format(value))
    return value

def process_device(device_desc, aliases):
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

def fail_if_no_ip(ipaddr):
    if not ipaddr:
        raise CliError('Local IP-address could not be determined')

def create_server_thread(filename, address, port, content_type, single_req=False):
    thr = Thread(target=serve_file, args=(filename, address, port, content_type, single_req))
    thr.setDaemon(True)
    thr.start()
    return thr
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('-d', '--device', metavar='NAME_OR_IP', help='Select Chromecast device.')
@click.version_option(version=VERSION, prog_name=PROGRAM_NAME, message='%(prog)s v%(version)s, ' + __codename__ + '.')
@click.pass_context
def cli(ctx, device):
    device_from_config = ctx.obj['options'].get('device')
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
def cast(settings, video_url, subtitles, force_default, random_play, no_subs, no_playlist, ytdl_option, seek_to, block=False):
    controller = 'default' if force_default or ytdl_option else None
    playlist_playback = False
    st_thr = su_thr = subs = None
    (cst, stream) = setup_cast(settings['selected_device'], video_url=video_url, prep='app', controller=controller, ytdl_options=ytdl_option)
    media_is_image = stream.guessed_content_category == 'image'
    local_or_remote = 'local' if stream.is_local_file else 'remote'
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
                entry = random.randrange(0, stream.playlist_length)
            else:
                echo_warning('Playlist playback not possible, playing first video')
                entry = 0
            stream.set_playlist_entry(entry)
    if playlist_playback:
        click.echo('Casting remote playlist {}...'.format(video_url))
        video_id = stream.video_id or stream.playlist_all_ids[0]
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
def cast_site(settings, url):
    cst = setup_cast(settings['selected_device'], controller='dashcast', action='load_url', prep='app')
    click.echo('Casting {} on "{}"...'.format(url, cst.cc_name))
    cst.load_url(url)

@cli.command(short_help='Add a video to the queue (YouTube only).')
@click.argument('video_url', callback=process_url)
@click.option('-n', '--play-next', is_flag=True, help='Add video immediately after currently playing video.')
@click.pass_obj
def add(settings, video_url, play_next):
    (cst, stream) = setup_cast(settings['selected_device'], video_url=video_url, action='add', prep='control')
    if cst.name != stream.extractor or not (stream.is_remote_file or stream.is_playlist_with_active_entry):
        raise CliError('This url cannot be added to the queue')
    click.echo('Adding video id "{}" to the queue.'.format(stream.video_id))
    if play_next:
        cst.add_next(stream.video_id)
    else:
        cst.add(stream.video_id)

@cli.command(short_help='Remove a video from the queue (YouTube only).')
@click.argument('video_url', callback=process_url)
@click.pass_obj
def remove(settings, video_url):
    (cst, stream) = setup_cast(settings['selected_device'], video_url=video_url, action='remove', prep='control')
    if cst.name != stream.extractor or not stream.is_remote_file:
        raise CliError('This url cannot be removed from the queue')
    click.echo('Removing video id "{}" from the queue.'.format(stream.video_id))
    cst.remove(stream.video_id)

@cli.command(short_help='Clear the queue (YouTube only).')
@click.pass_obj
def clear(settings):
    cst = setup_cast(settings['selected_device'], action='clear', prep='control')
    cst.clear()

@cli.command(short_help='Pause a video.')
@click.pass_obj
def pause(settings):
    cst = setup_cast(settings['selected_device'], action='pause', prep='control')
    cst.pause()

@cli.command(short_help='Resume a video after it has been paused.')
@click.pass_obj
def play(settings):
    cst = setup_cast(settings['selected_device'], action='play', prep='control')
    cst.play()

@cli.command('play_toggle', short_help='Toggle between playing and paused state.')
@click.pass_obj
def play_toggle(settings):
    cst = setup_cast(settings['selected_device'], action='play_toggle', prep='control')
    cst.play_toggle()

@cli.command(short_help='Stop playing.')
@click.option('-f', '--force', is_flag=True, help='Launch dummy chromecast app before sending stop command (for devices that do not respond to stop command under certain circumstances).')
@click.pass_obj
def stop(settings, force):
    cst = setup_cast(settings['selected_device'])
    cst.kill(force=force)

@cli.command(short_help='Rewind a video by TIME duration.')
@click.argument('timedesc', type=CATT_TIME, required=False, default='30', metavar='TIME')
@click.pass_obj
def rewind(settings, timedesc):
    cst = setup_cast(settings['selected_device'], action='rewind', prep='control')
    cst.rewind(timedesc)

@cli.command(short_help='Fastforward a video by TIME duration.')
@click.argument('timedesc', type=CATT_TIME, required=False, default='30', metavar='TIME')
@click.pass_obj
def ffwd(settings, timedesc):
    cst = setup_cast(settings['selected_device'], action='ffwd', prep='control')
    cst.ffwd(timedesc)

@cli.command(short_help='Seek the video to TIME position.')
@click.argument('timedesc', type=CATT_TIME, metavar='TIME')
@click.pass_obj
def seek(settings, timedesc):
    cst = setup_cast(settings['selected_device'], action='seek', prep='control')
    cst.seek(timedesc)

@cli.command(short_help='Skip to end of content.')
@click.pass_obj
def skip(settings):
    cst = setup_cast(settings['selected_device'], action='skip', prep='control')
    cst.skip()

@cli.command(short_help='Set the volume to LVL [0-100].')
@click.argument('level', type=click.IntRange(0, 100), metavar='LVL')
@click.pass_obj
def volume(settings, level):
    cst = setup_cast(settings['selected_device'])
    cst.volume(level / 100.0)

@cli.command(short_help='Turn up volume by a DELTA increment.')
@click.argument('delta', type=click.IntRange(1, 100), required=False, default=10, metavar='DELTA')
@click.pass_obj
def volumeup(settings, delta):
    cst = setup_cast(settings['selected_device'])
    cst.volumeup(delta / 100.0)

@cli.command(short_help='Turn down volume by a DELTA increment.')
@click.argument('delta', type=click.IntRange(1, 100), required=False, default=10, metavar='DELTA')
@click.pass_obj
def volumedown(settings, delta):
    cst = setup_cast(settings['selected_device'])
    cst.volumedown(delta / 100.0)

@cli.command(short_help='Enable or disable mute on supported devices.')
@click.argument('muted', type=click.BOOL, required=False, default=True, metavar='MUTED')
@click.pass_obj
def volumemute(settings, muted):
    cst = setup_cast(settings['selected_device'])
    cst.volumemute(muted)

@cli.command(short_help='Show some information about the currently-playing video.')
@click.pass_obj
def status(settings):
    cst = setup_cast(settings['selected_device'], prep='info')
    echo_status(cst.cast_info)

@cli.command(short_help='Show complete information about the currently-playing video.')
@click.option('-j', '--json-output', is_flag=True, help='Output info as json.')
@click.pass_obj
def info(settings, json_output):
    try:
        cst = setup_cast(settings['selected_device'], prep='info')
    except CastError:
        if json_output:
            info = {}
        else:
            raise
    else:
        info = cst.info
    if json_output:
        echo_json(info)
    else:
        for (key, value) in info.items():
            click.echo('{}: {}'.format(key, value))

@cli.command(short_help='Scan the local network and show all Chromecasts and their IPs.')
@click.option('-j', '--json-output', is_flag=True, help='Output scan result as json.')
def scan(json_output):
    if not json_output:
        click.echo('Scanning Chromecasts...')
    devices = get_cast_infos()
    if json_output:
        echo_json({d.friendly_name: d._asdict() for d in devices})
    else:
        if not devices:
            raise CastError('No devices found')
        for device in devices:
            click.echo(f'{device.host} - {device.friendly_name} - {device.manufacturer} {device.model_name}')

@cli.command(short_help='Save the current state of the Chromecast for later use.')
@click.argument('path', type=click.Path(writable=True), callback=process_path, required=False)
@click.pass_obj
def save(settings, path):
    cst = setup_cast(settings['selected_device'], prep='control')
    if not cst.save_capability or cst.is_streaming_local_file:
        raise CliError('Saving state of this kind of content is not supported')
    elif cst.save_capability == 'partial':
        echo_warning('Please be advised that playlist data will not be saved')
    echo_status(cst.cast_info)
    if path and path.is_file():
        click.confirm('File already exists. Overwrite?', abort=True)
    click.echo('Saving...')
    if path:
        state = CastState(path, StateMode.ARBI)
        cc_name = '*'
    else:
        state = CastState(STATE_PATH, StateMode.CONF)
        cc_name = cst.cc_name
    state.set_data(cc_name, {'controller': cst.name, 'data': cst.cast_info})

@cli.command(short_help='Return Chromecast to saved state.')
@click.argument('path', type=click.Path(exists=True), callback=process_path, required=False)
@click.pass_obj
def restore(settings, path):
    if not path and (not STATE_PATH.is_file()):
        raise CliError('Save file in config dir has not been created')
    cst = setup_cast(settings['selected_device'])
    state = CastState(path or STATE_PATH, StateMode.READ)
    try:
        data = state.get_data(cst.cc_name if not path else None)
    except StateFileError:
        raise CliError('The chosen file is not a valid save file')
    if not data:
        raise CliError('No save data found for this device')
    echo_status(data['data'])
    click.echo('Restoring...')
    cst = setup_cast(settings['selected_device'], prep='app', controller=data['controller'])
    cst.restore(data['data'])

@cli.command('write_config', short_help='DEPRECATED: Please use "set_default".')
def write_config():
    raise CliError('DEPRECATED: Please use "set_default"')

@cli.command('set_default', short_help='Set the selected device as default.')
@click.pass_obj
def set_default(settings):
    config = readconfig()
    device = get_device_from_settings(settings)
    config['options']['device'] = device
    writeconfig(config)

@cli.command('del_default', short_help='Delete the default device.')
@click.pass_obj
def del_default(settings):
    config = readconfig()
    if 'device' not in config['options']:
        raise CliError('No default device is set, so none deleted')
    config['options'].pop('device')
    writeconfig(config)

@cli.command('set_alias', short_help='Set an alias name for the selected device (case-insensitive).')
@click.argument('name')
@click.pass_obj
def set_alias(settings, name):
    config = readconfig()
    device = get_device_from_settings(settings)
    old_alias = get_alias_from_config(config, device)
    if old_alias:
        config['aliases'].pop(old_alias)
    config['aliases'][name] = device
    writeconfig(config)

@cli.command('del_alias', short_help='Delete the alias name of the selected device.')
@click.pass_obj
def del_alias(settings):
    config = readconfig()
    device = get_device_from_settings(settings)
    alias = get_alias_from_config(config, device)
    if not alias:
        raise CliError('No alias exists for "{}", so none deleted'.format(device))
    config['aliases'].pop(alias)
    writeconfig(config)

def get_alias_from_config(config, device):
    try:
        return next((a for (a, d) in config['aliases'].items() if d == device))
    except StopIteration:
        return None

def get_device_from_settings(settings):
    device_desc = settings['selected_device']
    if not device_desc or not settings['selected_device_is_from_cli']:
        raise CliError('No device specified (must be explicitly specified with -d option)')
    is_ip = is_ipaddress(device_desc)
    if is_ip:
        found = cast_ip_exists(device_desc)
    else:
        found = device_desc in [d.friendly_name for d in get_cast_infos()]
    if not found:
        msg = 'No device found at {}' if is_ip else 'Specified device "{}" not found'
        raise CliError(msg.format(device_desc))
    return device_desc

def writeconfig(config):
    try:
        CONFIG_DIR.mkdir(parents=True)
    except FileExistsError:
        pass
    with CONFIG_PATH.open('w') as configfile:
        config.write(configfile)

def readconfig():
    config = configparser.ConfigParser()
    config.read(str(CONFIG_PATH))
    for req_section in ('options', 'aliases'):
        if req_section not in config.sections():
            config.add_section(req_section)
    return config

def get_config_as_dict():
    """
    Returns a dictionary of the form:
        {"options": {"key": "value"},
         "aliases": {"device1": "device_name"}}
    """
    config = readconfig()
    return {section: dict(config.items(section)) for section in config.sections()}

def main():
    try:
        return cli(obj=get_config_as_dict())
    except CattUserError as err:
        sys.exit('Error: {}.'.format(str(err)))
if __name__ == '__main__':
    main()

# File: catt\cli_gpt4o.py

import configparser
import random
import sys
import time
from typing import Optional, Dict, Tuple, Any, List
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
CONFIG_DIR = Path(click.get_app_dir('catt'))
CONFIG_PATH = Path(CONFIG_DIR, 'catt.cfg')
STATE_PATH = Path(CONFIG_DIR, 'state.json')
WAIT_PLAY_TIMEOUT = 30
PROGRAM_NAME = 'catt'
try:
    VERSION = version(PROGRAM_NAME)
except Exception:
    VERSION = '0.0.0u'

class CattTimeParamType(click.ParamType):

    def convert(self, value, param, ctx):
        try:
            tdesc = [int(x) for x in value.split(':')]
            tlen = len(tdesc)
            if tlen > 1 and any((t > 59 for t in tdesc)) or tlen > 3:
                raise ValueError
        except ValueError:
            self.fail('{} is not a valid time description.'.format(value))
        tdesc.reverse()
        return sum((tdesc[p] * 60 ** p for p in range(tlen)))
CATT_TIME = CattTimeParamType()

class YtdlOptParamType(click.ParamType):

    def convert(self, value, param, ctx):
        if '=' not in value:
            self.fail('{} is not a valid key/value pair.'.format(value))
        (ykey, yval) = value.split('=', 1)
        yval = {'true': True, 'false': False}.get(yval.lower().strip(), yval)
        return (ykey, yval)
YTDL_OPT = YtdlOptParamType()

def process_url(ctx, param, value):
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

def process_path(ctx, param, value):
    path = Path(value) if value else None
    if path and (path.is_dir() or not path.parent.exists()):
        raise CliError('The specified path is invalid')
    return path

def process_subtitles(ctx, param, value):
    if not value:
        return None
    pval = urlparse(value).path if '://' in value else value
    if not pval.lower().endswith(('.srt', '.vtt')):
        raise CliError('Invalid subtitles format, only srt and vtt are supported')
    if '://' not in value and (not Path(value).is_file()):
        raise CliError('Subtitles file [{}] does not exist'.format(value))
    return value

def process_device(device_desc, aliases):
    if is_ipaddress(device_desc):
        return device_desc
    elif device_desc:
        return aliases.get(device_desc.lower(), device_desc)
    else:
        return device_desc

def fail_if_no_ip(ipaddr):
    if not ipaddr:
        raise CliError('Local IP-address could not be determined')

def create_server_thread(filename, address, port, content_type, single_req=False):
    thr = Thread(target=serve_file, args=(filename, address, port, content_type, single_req))
    thr.setDaemon(True)
    thr.start()
    return thr
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('-d', '--device', metavar='NAME_OR_IP', help='Select Chromecast device.')
@click.version_option(version=VERSION, prog_name=PROGRAM_NAME, message='%(prog)s v%(version)s, ' + __codename__ + '.')
@click.pass_context
def cli(ctx, device):
    device_from_config = ctx.obj['options'].get('device')
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
def cast(settings, video_url, subtitles, force_default, random_play, no_subs, no_playlist, ytdl_option, seek_to, block=False):
    controller = 'default' if force_default or ytdl_option else None
    playlist_playback = False
    st_thr = None
    su_thr = None
    subs = None
    (cst, stream) = setup_cast(settings['selected_device'], video_url=video_url, prep='app', controller=controller, ytdl_options=ytdl_option)
    media_is_image = stream.guessed_content_category == 'image'
    local_or_remote = 'local' if stream.is_local_file else 'remote'
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
                entry = random.randrange(0, stream.playlist_length)
            else:
                echo_warning('Playlist playback not possible, playing first video')
                entry = 0
            stream.set_playlist_entry(entry)
    if playlist_playback:
        click.echo('Casting remote playlist {}...'.format(video_url))
        video_id = stream.video_id or stream.playlist_all_ids[0]
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
def cast_site(settings, url):
    cst = setup_cast(settings['selected_device'], controller='dashcast', action='load_url', prep='app')
    click.echo('Casting {} on "{}"...'.format(url, cst.cc_name))
    cst.load_url(url)

@cli.command(short_help='Add a video to the queue (YouTube only).')
@click.argument('video_url', callback=process_url)
@click.option('-n', '--play-next', is_flag=True, help='Add video immediately after currently playing video.')
@click.pass_obj
def add(settings, video_url, play_next):
    (cst, stream) = setup_cast(settings['selected_device'], video_url=video_url, action='add', prep='control')
    if cst.name != stream.extractor or not (stream.is_remote_file or stream.is_playlist_with_active_entry):
        raise CliError('This url cannot be added to the queue')
    click.echo('Adding video id "{}" to the queue.'.format(stream.video_id))
    if play_next:
        cst.add_next(stream.video_id)
    else:
        cst.add(stream.video_id)

@cli.command(short_help='Remove a video from the queue (YouTube only).')
@click.argument('video_url', callback=process_url)
@click.pass_obj
def remove(settings, video_url):
    (cst, stream) = setup_cast(settings['selected_device'], video_url=video_url, action='remove', prep='control')
    if cst.name != stream.extractor or not stream.is_remote_file:
        raise CliError('This url cannot be removed from the queue')
    click.echo('Removing video id "{}" from the queue.'.format(stream.video_id))
    cst.remove(stream.video_id)

@cli.command(short_help='Clear the queue (YouTube only).')
@click.pass_obj
def clear(settings):
    cst = setup_cast(settings['selected_device'], action='clear', prep='control')
    cst.clear()

@cli.command(short_help='Pause a video.')
@click.pass_obj
def pause(settings):
    cst = setup_cast(settings['selected_device'], action='pause', prep='control')
    cst.pause()

@cli.command(short_help='Resume a video after it has been paused.')
@click.pass_obj
def play(settings):
    cst = setup_cast(settings['selected_device'], action='play', prep='control')
    cst.play()

@cli.command('play_toggle', short_help='Toggle between playing and paused state.')
@click.pass_obj
def play_toggle(settings):
    cst = setup_cast(settings['selected_device'], action='play_toggle', prep='control')
    cst.play_toggle()

@cli.command(short_help='Stop playing.')
@click.option('-f', '--force', is_flag=True, help='Launch dummy chromecast app before sending stop command (for devices that do not respond to stop command under certain circumstances).')
@click.pass_obj
def stop(settings, force):
    cst = setup_cast(settings['selected_device'])
    cst.kill(force=force)

@cli.command(short_help='Rewind a video by TIME duration.')
@click.argument('timedesc', type=CATT_TIME, required=False, default='30', metavar='TIME')
@click.pass_obj
def rewind(settings, timedesc):
    cst = setup_cast(settings['selected_device'], action='rewind', prep='control')
    cst.rewind(timedesc)

@cli.command(short_help='Fastforward a video by TIME duration.')
@click.argument('timedesc', type=CATT_TIME, required=False, default='30', metavar='TIME')
@click.pass_obj
def ffwd(settings, timedesc):
    cst = setup_cast(settings['selected_device'], action='ffwd', prep='control')
    cst.ffwd(timedesc)

@cli.command(short_help='Seek the video to TIME position.')
@click.argument('timedesc', type=CATT_TIME, metavar='TIME')
@click.pass_obj
def seek(settings, timedesc):
    cst = setup_cast(settings['selected_device'], action='seek', prep='control')
    cst.seek(timedesc)

@cli.command(short_help='Skip to end of content.')
@click.pass_obj
def skip(settings):
    cst = setup_cast(settings['selected_device'], action='skip', prep='control')
    cst.skip()

@cli.command(short_help='Set the volume to LVL [0-100].')
@click.argument('level', type=click.IntRange(0, 100), metavar='LVL')
@click.pass_obj
def volume(settings, level):
    cst = setup_cast(settings['selected_device'])
    cst.volume(level / 100.0)

@cli.command(short_help='Turn up volume by a DELTA increment.')
@click.argument('delta', type=click.IntRange(1, 100), required=False, default=10, metavar='DELTA')
@click.pass_obj
def volumeup(settings, delta):
    cst = setup_cast(settings['selected_device'])
    cst.volumeup(delta / 100.0)

@cli.command(short_help='Turn down volume by a DELTA increment.')
@click.argument('delta', type=click.IntRange(1, 100), required=False, default=10, metavar='DELTA')
@click.pass_obj
def volumedown(settings, delta):
    cst = setup_cast(settings['selected_device'])
    cst.volumedown(delta / 100.0)

@cli.command(short_help='Enable or disable mute on supported devices.')
@click.argument('muted', type=click.BOOL, required=False, default=True, metavar='MUTED')
@click.pass_obj
def volumemute(settings, muted):
    cst = setup_cast(settings['selected_device'])
    cst.volumemute(muted)

@cli.command(short_help='Show some information about the currently-playing video.')
@click.pass_obj
def status(settings):
    cst = setup_cast(settings['selected_device'], prep='info')
    echo_status(cst.cast_info)

@cli.command(short_help='Show complete information about the currently-playing video.')
@click.option('-j', '--json-output', is_flag=True, help='Output info as json.')
@click.pass_obj
def info(settings, json_output):
    try:
        cst = setup_cast(settings['selected_device'], prep='info')
    except CastError:
        if json_output:
            info = {}
        else:
            raise
    else:
        info = cst.info
    if json_output:
        echo_json(info)
    else:
        for (key, value) in info.items():
            click.echo('{}: {}'.format(key, value))

@cli.command(short_help='Scan the local network and show all Chromecasts and their IPs.')
@click.option('-j', '--json-output', is_flag=True, help='Output scan result as json.')
def scan(json_output):
    if not json_output:
        click.echo('Scanning Chromecasts...')
    devices = get_cast_infos()
    if json_output:
        echo_json({d.friendly_name: d._asdict() for d in devices})
    else:
        if not devices:
            raise CastError('No devices found')
        for device in devices:
            click.echo(f'{device.host} - {device.friendly_name} - {device.manufacturer} {device.model_name}')

@cli.command(short_help='Save the current state of the Chromecast for later use.')
@click.argument('path', type=click.Path(writable=True), callback=process_path, required=False)
@click.pass_obj
def save(settings, path):
    cst = setup_cast(settings['selected_device'], prep='control')
    if not cst.save_capability or cst.is_streaming_local_file:
        raise CliError('Saving state of this kind of content is not supported')
    elif cst.save_capability == 'partial':
        echo_warning('Please be advised that playlist data will not be saved')
    echo_status(cst.cast_info)
    if path and path.is_file():
        click.confirm('File already exists. Overwrite?', abort=True)
    click.echo('Saving...')
    if path:
        state = CastState(path, StateMode.ARBI)
        cc_name = '*'
    else:
        state = CastState(STATE_PATH, StateMode.CONF)
        cc_name = cst.cc_name
    state.set_data(cc_name, {'controller': cst.name, 'data': cst.cast_info})

@cli.command(short_help='Return Chromecast to saved state.')
@click.argument('path', type=click.Path(exists=True), callback=process_path, required=False)
@click.pass_obj
def restore(settings, path):
    if not path and (not STATE_PATH.is_file()):
        raise CliError('Save file in config dir has not been created')
    cst = setup_cast(settings['selected_device'])
    state = CastState(path or STATE_PATH, StateMode.READ)
    try:
        data = state.get_data(cst.cc_name if not path else None)
    except StateFileError:
        raise CliError('The chosen file is not a valid save file')
    if not data:
        raise CliError('No save data found for this device')
    echo_status(data['data'])
    click.echo('Restoring...')
    cst = setup_cast(settings['selected_device'], prep='app', controller=data['controller'])
    cst.restore(data['data'])

@cli.command('write_config', short_help='DEPRECATED: Please use "set_default".')
def write_config():
    raise CliError('DEPRECATED: Please use "set_default"')

@cli.command('set_default', short_help='Set the selected device as default.')
@click.pass_obj
def set_default(settings):
    config = readconfig()
    device = get_device_from_settings(settings)
    config['options']['device'] = device
    writeconfig(config)

@cli.command('del_default', short_help='Delete the default device.')
@click.pass_obj
def del_default(settings):
    config = readconfig()
    if 'device' not in config['options']:
        raise CliError('No default device is set, so none deleted')
    config['options'].pop('device')
    writeconfig(config)

@cli.command('set_alias', short_help='Set an alias name for the selected device (case-insensitive).')
@click.argument('name')
@click.pass_obj
def set_alias(settings, name):
    config = readconfig()
    device = get_device_from_settings(settings)
    old_alias = get_alias_from_config(config, device)
    if old_alias:
        config['aliases'].pop(old_alias)
    config['aliases'][name] = device
    writeconfig(config)

@cli.command('del_alias', short_help='Delete the alias name of the selected device.')
@click.pass_obj
def del_alias(settings):
    config = readconfig()
    device = get_device_from_settings(settings)
    alias = get_alias_from_config(config, device)
    if not alias:
        raise CliError('No alias exists for "{}", so none deleted'.format(device))
    config['aliases'].pop(alias)
    writeconfig(config)

def get_alias_from_config(config, device):
    try:
        return next((a for (a, d) in config['aliases'].items() if d == device))
    except StopIteration:
        return None

def get_device_from_settings(settings):
    device_desc = settings['selected_device']
    if not device_desc or not settings['selected_device_is_from_cli']:
        raise CliError('No device specified (must be explicitly specified with -d option)')
    is_ip = is_ipaddress(device_desc)
    if is_ip:
        found = cast_ip_exists(device_desc)
    else:
        found = device_desc in [d.friendly_name for d in get_cast_infos()]
    if not found:
        msg = 'No device found at {}' if is_ip else 'Specified device "{}" not found'
        raise CliError(msg.format(device_desc))
    return device_desc

def writeconfig(config):
    try:
        CONFIG_DIR.mkdir(parents=True)
    except FileExistsError:
        pass
    with CONFIG_PATH.open('w') as configfile:
        config.write(configfile)

def readconfig():
    config = configparser.ConfigParser()
    config.read(str(CONFIG_PATH))
    for req_section in ('options', 'aliases'):
        if req_section not in config.sections():
            config.add_section(req_section)
    return config

def get_config_as_dict():
    config = readconfig()
    return {section: dict(config.items(section)) for section in config.sections()}

def main():
    try:
        return cli(obj=get_config_as_dict())
    except CattUserError as err:
        sys.exit('Error: {}.'.format(str(err)))
if __name__ == '__main__':
    main()

# File: catt\controllers.py

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
DEFAULT_APP = App(app_name='default', app_id=MEDIA_RECEIVER_APP_ID, supported_device_types=['cast', 'audio', 'group'])
APPS = [DEFAULT_APP, App(app_name='youtube', app_id=YOUTUBE_APP_ID, supported_device_types=['cast']), App(app_name='dashcast', app_id=DASHCAST_APP_ID, supported_device_types=['cast', 'audio'])]

def get_app(id_or_name, cast_type=None, strict=False, show_warning=False):
    try:
        app = next((a for a in APPS if id_or_name in [a.id, a.name]))
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

def get_controller(cast, app, action=None, prep=None):
    controller = {'youtube': YoutubeCastController, 'dashcast': DashCastController}.get(app.name, DefaultCastController)
    if action and action not in dir(controller):
        raise ControllerError('This action is not supported by the {} controller'.format(app.name))
    return controller(cast, app, prep=prep)

def setup_cast(device_desc, video_url=None, controller=None, ytdl_options=None, action=None, prep=None):
    cast = get_cast(device_desc)
    cast_type = cast.cast_type
    app_id = cast.app_id
    stream = StreamInfo(video_url, cast_info=cast.cast_info, ytdl_options=ytdl_options) if video_url else None
    if controller:
        app = get_app(controller, cast_type, strict=True)
    elif prep == 'app' and stream and (not stream.is_local_file):
        app = get_app(stream.extractor, cast_type, show_warning=True)
    elif prep == 'control':
        if not app_id or app_id == BACKDROP_APP_ID:
            raise CastError('Chromecast is inactive')
        app = get_app(app_id, cast_type)
    else:
        app = get_app('default')
    cast_controller = get_controller(cast, app, action=action, prep=prep)
    return (cast_controller, stream) if stream else cast_controller

class CattStore:

    def __init__(self, store_path):
        self.store_path = store_path

    def _create_store_dir(self):
        try:
            self.store_path.parent.mkdir()
        except FileExistsError:
            pass

    def _read_store(self):
        with self.store_path.open() as store:
            return json.load(store)

    def _write_store(self, data):
        with self.store_path.open('w') as store:
            json.dump(data, store)

    def get_data(self, *args):
        raise NotImplementedError

    def set_data(self, *args):
        raise NotImplementedError

    def clear(self):
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

    def get_data(self, name):
        try:
            data = self._read_store()
            if set(next(iter(data.values())).keys()) != set(['controller', 'data']):
                raise ValueError
        except (json.decoder.JSONDecodeError, ValueError, StopIteration, AttributeError):
            raise StateFileError
        if name:
            return data.get(name)
        else:
            return next(iter(data.values()))

    def set_data(self, name, value):
        data = self._read_store()
        data[name] = value
        self._write_store(data)

class CastStatusListener:

    def __init__(self, app_id, active_app_id=None):
        self.app_id = app_id
        self.app_ready = threading.Event()
        if (active_app_id and app_id == active_app_id) and app_id != DASHCAST_APP_ID:
            self.app_ready.set()

    def new_cast_status(self, status):
        if self._is_app_ready(status):
            self.app_ready.set()
        else:
            self.app_ready.clear()

    def _is_app_ready(self, status):
        if status.app_id == self.app_id == DASHCAST_APP_ID:
            return status.status_text == 'Application ready'
        return status.app_id == self.app_id

class MediaStatusListener:

    def __init__(self, current_state, states, invert=False):
        if any((s not in VALID_STATE_EVENTS for s in states)):
            raise ListenerError('Invalid state(s)')
        if invert:
            self._states_waited_for = [s for s in VALID_STATE_EVENTS if s not in states]
        else:
            self._states_waited_for = states
        self._state_event = threading.Event()
        self._current_state = current_state
        if self._current_state in self._states_waited_for:
            self._state_event.set()

    def new_media_status(self, status):
        self._current_state = status.player_state
        if self._current_state in self._states_waited_for:
            self._state_event.set()
        else:
            self._state_event.clear()

    def wait_for_states(self, timeout=None):
        return self._state_event.wait(timeout=timeout)

class SimpleListener:

    def __init__(self):
        self._status_received = threading.Event()

    def new_media_status(self, status):
        self._status_received.set()

    def block_until_status_received(self):
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

    def prep_app(self):
        """Make sure desired chromecast app is running."""
        if not self._cast_listener.app_ready.is_set():
            self._cast.start_app(self._cast_listener.app_id)
            self._cast_listener.app_ready.wait()

    def prep_control(self):
        """Make sure chromecast is not idle."""
        self._update_status()
        if self._is_idle:
            raise CastError('Nothing is currently playing')

    def prep_info(self):
        self._update_status()

    def _update_status(self):

        def update():
            listener = SimpleListener()
            self._cast.media_controller.register_status_listener(listener)
            self._cast.media_controller.update_status()
            listener.block_until_status_received()
        if not self._supports_google_media_namespace:
            return
        update()
        status = self._cast.media_controller.status
        if status.current_time and (not status.content_id):
            update()

    @property
    def cc_name(self):
        return self._cast.cast_info.friendly_name

    @property
    def info(self):
        status = self._cast.media_controller.status.__dict__
        status.update(self._cast.status._asdict())
        return status

    @property
    def media_info(self):
        status = self._cast.media_controller.status
        return {'title': status.title, 'content_id': status.content_id, 'current_time': status.current_time if self._is_seekable else None, 'thumb': status.images[0].url if status.images else None}

    @property
    def cast_info(self):
        cinfo = {'volume_level': str(int(round(self._cast.status.volume_level, 2) * 100)), 'volume_muted': self._cast.status.volume_muted}
        if self._is_idle:
            return cinfo
        cinfo.update(self.media_info)
        status = self._cast.media_controller.status
        if self._is_seekable:
            (duration, current) = (status.duration, status.current_time)
            remaining = duration - current
            progress = int(1.0 * current / duration * 100)
            cinfo.update({'duration': duration, 'remaining': remaining, 'progress': progress})
        if self._is_audiovideo:
            cinfo.update({'player_state': status.player_state})
        return cinfo

    @property
    def is_streaming_local_file(self):
        status = self._cast.media_controller.status
        return status.content_id.endswith('?loaded_from_catt')

    @property
    def _supports_google_media_namespace(self):
        return GOOGLE_MEDIA_NAMESPACE in self._cast.status.namespaces

    @property
    def _is_seekable(self):
        status = self._cast.media_controller.status
        return status.duration and status.stream_type == 'BUFFERED'

    @property
    def _is_audiovideo(self):
        status = self._cast.media_controller.status
        content_type = status.content_type.split('/')[0] if status.content_type else None
        return content_type != 'image' if content_type else False

    @property
    def _is_idle(self):
        status = self._cast.media_controller.status
        app_id = self._cast.app_id
        return not app_id or app_id == BACKDROP_APP_ID or (status.player_state in ['UNKNOWN', 'IDLE'] and self._supports_google_media_namespace)

    def volume(self, level):
        self._cast.set_volume(level)

    def volumeup(self, delta):
        self._cast.volume_up(delta)

    def volumedown(self, delta):
        self._cast.volume_down(delta)

    def volumemute(self, muted):
        self._cast.set_volume_muted(muted)

    def kill(self, idle_only=False, force=False):
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
        if idle_only and (not self._is_idle):
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

    def play(self):
        self._cast.media_controller.play()

    def pause(self):
        self._cast.media_controller.pause()

    def play_toggle(self):
        state = self._cast.media_controller.status.player_state
        if state == 'PAUSED':
            self.play()
        elif state in ['BUFFERING', 'PLAYING']:
            self.pause()
        else:
            raise ValueError('Invalid or undefined state type')

    def seek(self, seconds):
        if self._is_seekable:
            self._cast.media_controller.seek(seconds)
        else:
            raise CastError('Stream is not seekable')

    def rewind(self, seconds):
        pos = self._cast.media_controller.status.current_time
        self.seek(pos - seconds)

    def ffwd(self, seconds):
        pos = self._cast.media_controller.status.current_time
        self.seek(pos + seconds)

    def skip(self):
        if self._is_seekable:
            self._cast.media_controller.skip()
        else:
            raise CastError('Stream is not skippable')

class PlaybackBaseMixin:
    _cast = None

    def play_media_url(self, video_url, **kwargs):
        raise NotImplementedError

    def play_media_id(self, video_id, **kwargs):
        raise NotImplementedError

    def play_playlist(self, playlist_id, video_id):
        raise NotImplementedError

    def wait_for(self, states, invert=False, timeout=None):
        media_listener = MediaStatusListener(self._cast.media_controller.status.player_state, states, invert=invert)
        self._cast.media_controller.register_status_listener(media_listener)
        try:
            return media_listener.wait_for_states(timeout=timeout)
        except pychromecast.error.UnsupportedNamespace:
            raise CastError('Chromecast app operation was interrupted')

    def restore(self, data):
        raise NotImplementedError

class DefaultCastController(CastController, MediaControllerMixin, PlaybackBaseMixin):

    def __init__(self, cast, app, prep=None):
        super(DefaultCastController, self).__init__(cast, app, prep=prep)
        self.info_type = 'url'
        self.save_capability = 'complete' if self._is_seekable and self._cast.app_id == DEFAULT_APP.id else None

    def play_media_url(self, video_url, **kwargs):
        content_type = kwargs.get('content_type') or 'video/mp4'
        self._controller.play_media(video_url, content_type, current_time=kwargs.get('current_time'), title=kwargs.get('title'), thumb=kwargs.get('thumb'), subtitles=kwargs.get('subtitles'))
        self._controller.block_until_active()

    def restore(self, data):
        self.play_media_url(data['content_id'], current_time=data['current_time'], title=data['title'], thumb=data['thumb'])

class DashCastController(CastController):

    def __init__(self, cast, app, prep=None):
        self._controller = PyChromecastDashCastController()
        super(DashCastController, self).__init__(cast, app, prep=prep)

    def load_url(self, url, **kwargs):
        self._controller.load_url(url, force=True)

    def prep_app(self):
        """Make sure desired chromecast app is running."""
        self._cast.start_app(self._cast_listener.app_id, force_launch=True)
        self._cast_listener.app_ready.wait()

class YoutubeCastController(CastController, MediaControllerMixin, PlaybackBaseMixin):

    def __init__(self, cast, app, prep=None):
        self._controller = YouTubeController()
        super(YoutubeCastController, self).__init__(cast, app, prep=prep)
        self.info_type = 'id'
        self.save_capability = 'partial'
        self.playlist_capability = 'complete'

    def play_media_id(self, video_id, **kwargs):
        self._controller.play_video(video_id)
        current_time = kwargs.get('current_time')
        if current_time:
            self.wait_for(['PLAYING'])
            self.seek(current_time)

    def play_playlist(self, playlist_id, video_id):
        self.clear()
        self._controller.play_video(video_id, playlist_id)

    def add(self, video_id):
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.add_to_queue(video_id)

    def add_next(self, video_id):
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.play_next(video_id)

    def remove(self, video_id):
        self.wait_for(['BUFFERING'], invert=True)
        self._controller.remove_video(video_id)

    def clear(self):
        self._controller.clear_playlist()

    def restore(self, data):
        self.play_media_id(data['content_id'], current_time=data['current_time'])

# File: catt\discovery.py

from typing import List
from typing import Optional
from typing import Union
import pychromecast
from .error import CastError
from .util import is_ipaddress
DEFAULT_PORT = 8009

def get_casts(names=None):
    """
    Discover all available devices, optionally filtering them with list of specific device names
    (which will speedup discovery, as pychromecast does this in a non-blocking manner).

    :param names: Optional list of device names.
    :type names: List[str]
    :returns: List of Chromecast objects.
    :rtype: List[pychromecast.Chromecast]
    """
    if names:
        (cast_infos, browser) = pychromecast.discovery.discover_listed_chromecasts(friendly_names=names)
    else:
        (cast_infos, browser) = pychromecast.discovery.discover_chromecasts()
    casts = [pychromecast.get_chromecast_from_cast_info(c, browser.zc) for c in cast_infos]
    for cast in casts:
        cast.wait()
    browser.stop_discovery()
    casts.sort(key=lambda c: c.cast_info.friendly_name)
    return casts

def get_cast_infos():
    """
    Discover all available devices, and collect info from them.

    :returns: List of CastInfo namedtuples.
    :rtype: List[pychromecast.CastInfo]
    """
    return [c.cast_info for c in get_casts()]

def get_cast_with_name(cast_name):
    """
    Get specific device if supplied name is not None,
    otherwise the device with the name that has the lowest alphabetical value.

    :param device_name: Name of device.
    :type device_name: str
    :returns: Chromecast object.
    :rtype: pychromecast.Chromecast
    """
    casts = get_casts([cast_name]) if cast_name else get_casts()
    return casts[0] if casts else None

def get_cast_with_ip(cast_ip, port=DEFAULT_PORT):
    """
    Get specific device using its ip-address (and optionally port).

    :param device_ip: Ip-address of device.
    :type device_name: str
    :param port: Optional port number of device.
    :returns: Chromecast object.
    :rtype: pychromecast.Chromecast
    """
    device_info = pychromecast.discovery.get_device_info(cast_ip)
    if not device_info:
        return None
    host = (cast_ip, DEFAULT_PORT, device_info.uuid, device_info.model_name, device_info.friendly_name)
    cast = pychromecast.get_chromecast_from_host(host)
    cast.wait()
    return cast

def cast_ip_exists(cast_ip):
    """
    Get availability of specific device using its ip-address.

    :param device_ip: Ip-address of device.
    :type device_name: str
    :returns: Availability of device.
    :rtype: bool
    """
    return bool(get_cast_with_ip(cast_ip))

def get_cast(cast_desc=None):
    """
    Attempt to connect with requested device (or any device if none has been specified).

    :param device_desc: Can be an ip-address or a name.
    :type device_desc: str
    :returns: Chromecast object for use in a CastController.
    :rtype: pychromecast.Chromecast
    """
    cast = None
    if cast_desc and is_ipaddress(cast_desc):
        cast = get_cast_with_ip(cast_desc)
        if not cast:
            msg = 'No device found at {}'.format(cast_desc)
            raise CastError(msg)
    else:
        cast = get_cast_with_name(cast_desc)
        if not cast:
            msg = 'Specified device "{}" not found'.format(cast_desc) if cast_desc else 'No devices found'
            raise CastError(msg)
    return cast

# File: catt\discovery_gpt4o.py

from typing import List, Optional, Union
import pychromecast
from .error import CastError
from .util import is_ipaddress
DEFAULT_PORT = 8009

def get_casts(names=None):
    if names:
        (cast_infos, browser) = pychromecast.discovery.discover_listed_chromecasts(friendly_names=names)
    else:
        (cast_infos, browser) = pychromecast.discovery.discover_chromecasts()
    casts = [pychromecast.get_chromecast_from_cast_info(c, browser.zc) for c in cast_infos]
    for cast in casts:
        cast.wait()
    browser.stop_discovery()
    casts.sort(key=lambda c: c.cast_info.friendly_name)
    return casts

def get_cast_infos():
    return [c.cast_info for c in get_casts()]

def get_cast_with_name(cast_name):
    casts = get_casts([cast_name]) if cast_name else get_casts()
    return casts[0] if casts else None

def get_cast_with_ip(cast_ip, port=DEFAULT_PORT):
    device_info = pychromecast.discovery.get_device_info(cast_ip)
    if not device_info:
        return None
    host = (cast_ip, DEFAULT_PORT, device_info.uuid, device_info.model_name, device_info.friendly_name)
    cast = pychromecast.get_chromecast_from_host(host)
    cast.wait()
    return cast

def cast_ip_exists(cast_ip):
    return bool(get_cast_with_ip(cast_ip))

def get_cast(cast_desc=None):
    cast = None
    if cast_desc and is_ipaddress(cast_desc):
        cast = get_cast_with_ip(cast_desc)
        if not cast:
            msg = 'No device found at {}'.format(cast_desc)
            raise CastError(msg)
    else:
        cast = get_cast_with_name(cast_desc)
        if not cast:
            msg = 'Specified device "{}" not found'.format(cast_desc) if cast_desc else 'No devices found'
            raise CastError(msg)
    return cast

# File: catt\error.py

class CattError(Exception):
    """Base exception for catt."""
    pass

class CattUserError(CattError):
    """
    Messages from exceptions that inherit from this class,
    are transformed into error messages to the cli user.
    """
    pass

class StateFileError(CattError):
    """When a requested state file contains invalid data or nothing."""
    pass

class ListenerError(CattError):
    """When invalid data is passed to a listener class initializer."""
    pass

class AppSelectionError(CattError):
    """When invalid data is passed to the app selection mechanism."""
    pass

class PlaylistError(CattError):
    """When playlist specific operations are attempted with non-playlist info."""
    pass

class APIError(CattError):
    pass

class SubtitlesError(CattUserError):
    """When a specified subtitles file cannot be found or its encoding cannot be determined."""
    pass

class CliError(CattUserError):
    """When the cli user passes invalid commands/options/arguments to catt."""
    pass

class CastError(CattUserError):
    """When operations are attempted with non-existent or inactive devices."""
    pass

class ControllerError(CattUserError):
    """When a controller is incapable of the requested action."""
    pass

class ExtractionError(CattUserError):
    """When the requested media cannot be found or processed by yt-dlp."""
    pass

class FormatError(CattUserError):
    """When the supplied format filter is invalid or excludes all available formats."""
    pass

# File: catt\http_server.py

import io
import re
import socketserver
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from typing import Tuple
BYTE_RANGE_RE = re.compile('bytes=(\\d+)-(\\d+)?$')

def copy_byte_range(infile, outfile, start=None, stop=None, bufsize=16 * 1024):
    """Like shutil.copyfileobj, but only copy a range of the streams.

    Both start and stop are inclusive.
    """
    if start is not None:
        infile.seek(start)
    while True:
        to_read = min(bufsize, stop + 1 - infile.tell() if stop else bufsize)
        buf = infile.read(to_read)
        if not buf:
            break
        outfile.write(buf)

def parse_byte_range(byte_range):
    """Returns the two numbers in 'bytes=123-456' or throws ValueError.

    The last number or both numbers may be None.
    """
    if byte_range.strip() == '':
        return (None, None)
    match = BYTE_RANGE_RE.match(byte_range)
    if not match:
        raise ValueError('Invalid byte range {}'.format(byte_range))
    (first, last) = [int(x) if x else None for x in match.groups()]
    assert first is not None
    if last and last < first:
        raise ValueError('Invalid byte range {}'.format(byte_range))
    return (first, last)

def serve_file(filename, address='', port=45114, content_type=None, single_req=False):

    class FileHandler(BaseHTTPRequestHandler):

        def format_size(self, size):
            for size_unity in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024:
                    return (size, size_unity)
                size = size / 1024
            return (size * 1024, size_unity)

        def log_message(self, format, *args, **kwargs):
            (size, size_unity) = self.format_size(stats.st_size)
            format += ' {} - {:0.2f} {}'.format(content_type, size, size_unity)
            return super(FileHandler, self).log_message(format, *args, **kwargs)

        def do_GET(self):
            if 'Range' not in self.headers:
                (first, last) = (0, stats.st_size)
            else:
                try:
                    (first, last) = parse_byte_range(self.headers['Range'])
                except ValueError:
                    self.send_error(400, 'Invalid byte range')
                    return None
            if last is None or last >= stats.st_size:
                last = stats.st_size - 1
            response_length = last - first + 1
            try:
                if 'Range' not in self.headers:
                    self.send_response(200)
                else:
                    self.send_response(206)
                    self.send_header('Content-Range', 'bytes {}-{}/{}'.format(first, last, stats.st_size))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-type', content_type)
                self.send_header('Content-Length', str(response_length))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Last-Modified', time.strftime('%a %d %b %Y %H:%M:%S GMT', time.localtime(stats.st_mtime)))
                self.end_headers()
                mediafile = open(str(mediapath), 'rb')
                copy_byte_range(mediafile, self.wfile, first, last)
            except ConnectionResetError:
                pass
            except BrokenPipeError:
                print('Device disconnected while playing. Please check that the video file is compatible with the device.', file=sys.stderr)
            except:
                traceback.print_exc()
            mediafile.close()
    if content_type is None:
        content_type = 'video/mp4'
    mediapath = Path(filename)
    stats = mediapath.stat()
    httpd = socketserver.TCPServer((address, port), FileHandler)
    if single_req:
        httpd.handle_request()
    else:
        httpd.serve_forever()
    httpd.server_close()

# File: catt\http_server_gpt4o.py

import io
import re
import socketserver
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Tuple, Union
BYTE_RANGE_RE = re.compile('bytes=(\\d+)-(\\d+)?$')

def copy_byte_range(infile, outfile, start=None, stop=None, bufsize=16 * 1024):
    """Like shutil.copyfileobj, but only copy a range of the streams.

    Both start and stop are inclusive.
    """
    if start is not None:
        infile.seek(start)
    while True:
        to_read = min(bufsize, stop + 1 - infile.tell() if stop else bufsize)
        buf = infile.read(to_read)
        if not buf:
            break
        outfile.write(buf)

def parse_byte_range(byte_range):
    """Returns the two numbers in 'bytes=123-456' or throws ValueError.

    The last number or both numbers may be None.
    """
    if byte_range.strip() == '':
        return (None, None)
    match = BYTE_RANGE_RE.match(byte_range)
    if not match:
        raise ValueError('Invalid byte range {}'.format(byte_range))
    (first, last) = [int(x) if x else None for x in match.groups()]
    assert first is not None
    if last and last < first:
        raise ValueError('Invalid byte range {}'.format(byte_range))
    return (first, last)

def serve_file(filename, address='', port=45114, content_type=None, single_req=False):

    class FileHandler(BaseHTTPRequestHandler):

        def format_size(self, size):
            for size_unity in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024:
                    return (size, size_unity)
                size = size / 1024
            return (size * 1024, size_unity)

        def log_message(self, format, *args, **kwargs):
            (size, size_unity) = self.format_size(stats.st_size)
            format += ' {} - {:0.2f} {}'.format(content_type, size, size_unity)
            return super(FileHandler, self).log_message(format, *args, **kwargs)

        def do_GET(self):
            if 'Range' not in self.headers:
                (first, last) = (0, stats.st_size)
            else:
                try:
                    (first, last) = parse_byte_range(self.headers['Range'])
                except ValueError:
                    self.send_error(400, 'Invalid byte range')
                    return None
            if last is None or last >= stats.st_size:
                last = stats.st_size - 1
            response_length = last - first + 1
            try:
                if 'Range' not in self.headers:
                    self.send_response(200)
                else:
                    self.send_response(206)
                    self.send_header('Content-Range', 'bytes {}-{}/{}'.format(first, last, stats.st_size))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-type', content_type)
                self.send_header('Content-Length', str(response_length))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Last-Modified', time.strftime('%a %d %b %Y %H:%M:%S GMT', time.localtime(stats.st_mtime)))
                self.end_headers()
                mediafile = open(str(mediapath), 'rb')
                copy_byte_range(mediafile, self.wfile, first, last)
            except ConnectionResetError:
                pass
            except BrokenPipeError:
                print('Device disconnected while playing. Please check that the video file is compatible with the device.', file=sys.stderr)
            except:
                traceback.print_exc()
            mediafile.close()
    if content_type is None:
        content_type = 'video/mp4'
    mediapath = Path(filename)
    stats = mediapath.stat()
    httpd = socketserver.TCPServer((address, port), FileHandler)
    if single_req:
        httpd.handle_request()
    else:
        httpd.serve_forever()
    httpd.server_close()

# File: catt\stream_info.py

import random
from pathlib import Path
import yt_dlp
from .error import ExtractionError
from .error import FormatError
from .error import PlaylistError
from .util import get_local_ip
from .util import guess_mime
AUDIO_DEVICE_TYPES = ['audio', 'group']
ULTRA_MODELS = [('Xiaomi', 'MIBOX3'), ('Unknown manufacturer', 'Chromecast Ultra')]
BEST_MAX_2K = 'best[width <=? 1920][height <=? 1080]'
BEST_MAX_4K = 'best[width <=? 3840][height <=? 2160]'
BEST_ONLY_AUDIO = 'bestaudio'
BEST_FALLBACK = '/best'
MAX_50FPS = '[fps <=? 50]'
TWITCH_NO_60FPS = '[format_id != 1080p60__source_][format_id != 720p60]'
MIXCLOUD_NO_DASH_HLS = '[format_id != dash-a1-x3][format_id !*= hls-6]'
BANDCAMP_NO_AIFF_ALAC = '[format_id != aiff-lossless][format_id != alac]'
AUDIO_FORMAT = BEST_ONLY_AUDIO + MIXCLOUD_NO_DASH_HLS + BANDCAMP_NO_AIFF_ALAC + BEST_FALLBACK
ULTRA_FORMAT = BEST_MAX_4K + BANDCAMP_NO_AIFF_ALAC
STANDARD_FORMAT = BEST_MAX_2K + MAX_50FPS + TWITCH_NO_60FPS + BANDCAMP_NO_AIFF_ALAC
DEFAULT_YTDL_OPTS = {'quiet': True, 'no_warnings': True}

class StreamInfo:

    def __init__(self, video_url, cast_info=None, ytdl_options=None, throw_ytdl_dl_errs=False):
        self._throw_ytdl_dl_errs = throw_ytdl_dl_errs
        self.local_ip = get_local_ip(cast_info.host) if cast_info else None
        self.port = random.randrange(45000, 47000) if cast_info else None
        if '://' in video_url:
            self._ydl = yt_dlp.YoutubeDL(dict(ytdl_options) if ytdl_options else DEFAULT_YTDL_OPTS)
            self._preinfo = self._get_stream_preinfo(video_url)
            if self._preinfo.get('ie_key'):
                self._preinfo = self._get_stream_preinfo(self._preinfo['url'])
            self.is_local_file = False
            model = (cast_info.manufacturer, cast_info.model_name) if cast_info else None
            cast_type = cast_info.cast_type if cast_info else None
            if 'format' in self._ydl.params:
                self._best_format = self._ydl.params.pop('format')
            elif cast_type and cast_type in AUDIO_DEVICE_TYPES:
                self._best_format = AUDIO_FORMAT
            elif model and model in ULTRA_MODELS:
                self._best_format = ULTRA_FORMAT
            else:
                self._best_format = STANDARD_FORMAT
            if self.is_playlist:
                self._entries = list(self._preinfo['entries'])
                self._ydl.params.update({'noplaylist': True})
                vpreinfo = self._get_stream_preinfo(video_url)
                self._info = self._get_stream_info(vpreinfo) if 'entries' not in vpreinfo else None
            else:
                self._info = self._get_stream_info(self._preinfo)
        else:
            self._local_file = video_url
            self.is_local_file = True

    @property
    def is_remote_file(self):
        return not self.is_local_file and (not self.is_playlist)

    @property
    def _is_direct_link(self):
        return self.is_remote_file and self._info.get('direct')

    @property
    def is_playlist(self):
        return not self.is_local_file and 'entries' in self._preinfo

    @property
    def is_playlist_with_active_entry(self):
        return self.is_playlist and self._info

    @property
    def extractor(self):
        return self._preinfo['extractor'].split(':')[0] if not self.is_local_file else None

    @property
    def video_title(self):
        if self.is_local_file:
            return Path(self._local_file).stem
        elif self._is_direct_link:
            return Path(self._preinfo['webpage_url_basename']).stem
        elif self.is_remote_file or self.is_playlist_with_active_entry:
            return self._info['title']
        else:
            return None

    @property
    def video_url(self):
        if self.is_local_file:
            return 'http://{}:{}/?loaded_from_catt'.format(self.local_ip, self.port)
        elif self.is_remote_file or self.is_playlist_with_active_entry:
            return self._get_stream_url(self._info)
        else:
            return None

    @property
    def video_id(self):
        return self._info['id'] if self.is_remote_file or self.is_playlist_with_active_entry else None

    @property
    def video_thumbnail(self):
        return self._info.get('thumbnail') if self.is_remote_file or self.is_playlist_with_active_entry else None

    @property
    def guessed_content_type(self):
        if self.is_local_file:
            return guess_mime(Path(self._local_file).name)
        elif self._is_direct_link:
            return guess_mime(self._info['webpage_url_basename'])
        else:
            return None

    @property
    def guessed_content_category(self):
        content_type = self.guessed_content_type
        return content_type.split('/')[0] if content_type else None

    @property
    def playlist_length(self):
        return len(self._entries) if self.is_playlist else None

    @property
    def playlist_all_ids(self):
        if self.is_playlist and self._entries and self._entries[0].get('id'):
            return [entry['id'] for entry in self._entries]
        else:
            return None

    @property
    def playlist_title(self):
        return self._preinfo['title'] if self.is_playlist else None

    @property
    def playlist_id(self):
        return self._preinfo['id'] if self.is_playlist else None

    def set_playlist_entry(self, number):
        if self.is_playlist:
            if self._entries[number].get('ie_key'):
                entry = self._get_stream_preinfo(self._entries[number]['url'])
            else:
                entry = self._entries[number]
            self._info = self._get_stream_info(entry)
        else:
            raise PlaylistError('Called on non-playlist')

    def _get_stream_preinfo(self, video_url):
        try:
            return self._ydl.extract_info(video_url, process=False)
        except yt_dlp.utils.DownloadError:
            if self._throw_ytdl_dl_errs:
                raise
            else:
                raise ExtractionError('Remote resource not found')

    def _get_stream_info(self, preinfo):
        try:
            return self._ydl.process_ie_result(preinfo, download=False)
        except (yt_dlp.utils.ExtractorError, yt_dlp.utils.DownloadError):
            raise ExtractionError('yt-dlp extractor failed')

    def _get_stream_url(self, info):
        try:
            format_selector = self._ydl.build_format_selector(self._best_format)
        except ValueError:
            raise FormatError('The specified format filter is invalid')
        info.setdefault('incomplete_formats', {})
        try:
            best_format = next(format_selector(info))
        except StopIteration:
            raise FormatError('No suitable format was found')
        except KeyError:
            best_format = info
        return best_format['url']

# File: catt\subs_info.py

import re
import requests
from .error import SubtitlesError
from .util import create_temp_file

class SubsInfo:
    """
    This class facilitates fetching/reading a remote/local subtitles file,
    converting it to webvtt if needed, and then exposing a path to a tempfile
    holding the subtitles, ready to be served.
    An url to the (expected to be) served file is also exposed. The supplied
    local_ip and port params are used for this purpose.
    """

    def __init__(self, subs_url, local_ip, port):
        self._subs_url = subs_url
        self.local_ip = local_ip
        self.port = port
        subs = self._read_subs(subs_url)
        ext = subs_url.lower().split('.')[-1]
        if ext == 'srt':
            subs = self._convert_srt_to_webvtt(subs)
        self.file = create_temp_file(subs)

    @property
    def url(self):
        return 'http://{}:{}/{}'.format(self.local_ip, self.port, self.file)

    def _read_subs(self, subs_url):
        if '://' in subs_url:
            return self._fetch_remote_subs(subs_url)
        else:
            return self._read_local_subs(subs_url)

    def _convert_srt_to_webvtt(self, content):
        content = re.sub('^(.*? \\-\\-\\> .*?)$', lambda m: m.group(1).replace(',', '.'), content, flags=re.MULTILINE)
        return 'WEBVTT\n\n' + content

    def _read_local_subs(self, filename):
        for possible_encoding in ['utf-8', 'iso-8859-15']:
            try:
                with open(filename, 'r', encoding=possible_encoding) as srtfile:
                    content = srtfile.read()
                    return content
            except UnicodeDecodeError:
                pass
        raise SubtitlesError('Could not find the proper encoding of {}. Please convert it to utf-8'.format(filename))

    def _fetch_remote_subs(self, url):
        response = requests.get(url)
        if not response:
            raise SubtitlesError('Remote subtitles file not found')
        return response.text

# File: catt\subs_info_gpt4o.py

import re
import requests
from typing import Union
from .error import SubtitlesError
from .util import create_temp_file

class SubsInfo:
    """
    This class facilitates fetching/reading a remote/local subtitles file,
    converting it to webvtt if needed, and then exposing a path to a tempfile
    holding the subtitles, ready to be served.
    An url to the (expected to be) served file is also exposed. The supplied
    local_ip and port params are used for this purpose.
    """

    def __init__(self, subs_url, local_ip, port):
        self._subs_url = subs_url
        self.local_ip = local_ip
        self.port = port
        subs = self._read_subs(subs_url)
        ext = subs_url.lower().split('.')[-1]
        if ext == 'srt':
            subs = self._convert_srt_to_webvtt(subs)
        self.file = create_temp_file(subs)

    @property
    def url(self):
        return 'http://{}:{}/{}'.format(self.local_ip, self.port, self.file)

    def _read_subs(self, subs_url):
        if '://' in subs_url:
            return self._fetch_remote_subs(subs_url)
        else:
            return self._read_local_subs(subs_url)

    def _convert_srt_to_webvtt(self, content):
        content = re.sub('^(.*? \\-\\-\\> .*?)$', lambda m: m.group(1).replace(',', '.'), content, flags=re.MULTILINE)
        return 'WEBVTT\n\n' + content

    def _read_local_subs(self, filename):
        for possible_encoding in ['utf-8', 'iso-8859-15']:
            try:
                with open(filename, 'r', encoding=possible_encoding) as srtfile:
                    content = srtfile.read()
                    return content
            except UnicodeDecodeError:
                pass
        raise SubtitlesError('Could not find the proper encoding of {}. Please convert it to utf-8'.format(filename))

    def _fetch_remote_subs(self, url):
        response = requests.get(url)
        if not response:
            raise SubtitlesError('Remote subtitles file not found')
        return response.text

# File: catt\util.py

import ipaddress
import json
import socket
import tempfile
import time
from pathlib import Path
import click
import ifaddr

def echo_warning(msg):
    click.secho('Warning: ', fg='red', nl=False, err=True)
    click.echo('{}.'.format(msg), err=True)

def echo_json(data_dict):
    click.echo(json.dumps(data_dict, indent=4, default=str))

def echo_status(status):
    if status.get('title'):
        click.echo('Title: {}'.format(status['title']))
    if status.get('current_time'):
        current = human_time(status['current_time'])
        if status.get('duration'):
            duration = human_time(status['duration'])
            remaining = human_time(status['remaining'])
            click.echo('Time: {} / {} ({}%)'.format(current, duration, status['progress']))
            click.echo('Remaining time: {}'.format(remaining))
        else:
            click.echo('Time: {}'.format(current))
    if status.get('player_state'):
        click.echo('State: {}'.format(status['player_state']))
    click.echo('Volume: {}'.format(status['volume_level']))
    click.echo('Volume muted: {}'.format(status['volume_muted']))

def guess_mime(path):
    extension = Path(path).suffix.lower()
    extensions = {'.mp4': 'video/mp4', '.m4a': 'audio/mp4', '.mp3': 'audio/mp3', '.mpa': 'audio/mpeg', '.webm': 'video/webm', '.mkv': 'video/x-matroska', '.bmp': 'image/bmp', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.png': 'image/png', '.webp': 'image/web'}
    return extensions.get(extension, 'video/mp4')

def hunt_subtitles(video):
    """Searches for subtitles in the current folder"""
    video_path = Path(video)
    video_path_stem_lower = video_path.stem.lower()
    for entry_path in video_path.parent.iterdir():
        if entry_path.is_dir():
            continue
        if entry_path.stem.lower().startswith(video_path_stem_lower) and entry_path.suffix.lower() in ['.vtt', '.srt']:
            return str(entry_path.resolve())
    return None

def create_temp_file(content):
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.vtt', delete=False) as tfile:
        tfile.write(content.encode())
        return tfile.name

def human_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def get_local_ip(host):
    """
    The primary ifaddr based approach, tries to guess the local ip from the cc ip,
    by comparing the subnet of ip-addresses of all the local adapters to the subnet of the cc ip.
    This should work on all platforms, but requires the catt box and the cc to be on the same subnet.
    As a fallback we use a socket based approach, that does not suffer from this limitation, but
    might not work on all platforms.
    """
    host_ipversion = type(ipaddress.ip_address(host))
    for adapter in ifaddr.get_adapters():
        for adapter_ip in adapter.ips:
            aip = adapter_ip.ip[0] if isinstance(adapter_ip.ip, tuple) else adapter_ip.ip
            try:
                if not isinstance(ipaddress.ip_address(aip), host_ipversion):
                    continue
            except ValueError:
                continue
            ipt = [(ip, adapter_ip.network_prefix) for ip in (aip, host)]
            (catt_net, cc_net) = [ipaddress.ip_network('{0}/{1}'.format(*ip), strict=False) for ip in ipt]
            if catt_net == cc_net:
                return aip
            else:
                continue
    try:
        return [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    except OSError:
        return None

def is_ipaddress(device):
    try:
        ipaddress.ip_address(device)
    except ValueError:
        return False
    else:
        return True

# File: examples\accidents.py

import sys
from catt.api import CattDevice
VIDEOS = ['https://www.youtube.com/watch?v=mt084vYqbnY', 'https://www.youtube.com/watch?v=INxcj8_Zlo8', 'https://www.youtube.com/watch?v=KDrpPqsXfVU']

def ouch(device):
    cast = CattDevice(name=device)
    for video in VIDEOS:
        cast.play_url(video, resolve=True, block=True)
        print('OOOOUUUUUUUUUUUUCCHHH!!!!!')
if __name__ == '__main__':
    ouch(sys.argv[1])

# File: realcc_tests\test_procedure.py

import json
import subprocess
import time
from typing import Any
import click
CMD_BASE = ['catt', '-d']
VALIDATE_ARGS = ['info', '-j']
STOP_ARGS = ['stop']
SCAN_CMD = ['catt', 'scan', '-j']

def subp_run(cmd, allow_failure=False):
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if not allow_failure and output.returncode != 0:
        raise CattTestError('The command "{}" failed.'.format(' '.join(cmd)))
    return output

class CattTestError(click.ClickException):
    pass

class CattTest:

    def __init__(self, desc, cmd_args, sleep=10, should_fail=False, substring=False, time_test=False, check_data=None, check_err=''):
        if should_fail and (not check_err) or (not should_fail and (not check_data)):
            raise CattTestError('Expected outcome mismatch.')
        if substring and time_test:
            raise CattTestError('Test type mismatch.')
        self._cmd_args = cmd_args
        self._cmd = []
        self._validate_cmd = []
        self._sleep = sleep
        self._should_fail = should_fail
        self._substring = substring
        self._time_test = time_test
        (self._check_key, self._check_val) = check_data if check_data else (None, None)
        self._check_err = check_err
        self._output = None
        self._failed = False
        self.desc = desc + (' (should fail)' if self._should_fail else '')
        self.dump = ''

    def set_cmd_base(self, base):
        self._cmd = base + self._cmd_args
        self._validate_cmd = base + VALIDATE_ARGS

    def _get_val(self, key):
        output = subp_run(self._validate_cmd)
        catt_json = json.loads(output.stdout)
        return catt_json[key]

    def _should_fail_test(self):
        if self._should_fail == self._failed:
            return True
        else:
            self.dump += self._output.stderr if self._failed else self._output.stdout
            return False

    def _failure_test(self):
        output_errmsg = self._output.stderr.splitlines()[-1]
        if output_errmsg == 'Error: {}.'.format(self._check_err):
            self.dump += '{}\n - The expected error message.'.format(output_errmsg)
            return True
        else:
            self.dump += self._output.stderr
            return False

    def _regular_test(self, time_margin=5):
        catt_val = self._get_val(self._check_key)
        if self._time_test:
            passed = abs(int(catt_val) - int(self._check_val)) <= time_margin
            extra_info = '(time margin is {} seconds)'.format(time_margin)
        elif self._substring:
            passed = self._check_val in catt_val
            extra_info = '(substring)'
        else:
            passed = catt_val == self._check_val
            extra_info = ''
        if not passed:
            self.dump += 'Expected data from "{}" key:\n{} {}\nActual data:\n{}'.format(self._check_key, self._check_val, extra_info, catt_val)
        return passed

    def run(self):
        self._output = subp_run(self._cmd, allow_failure=True)
        self._failed = self._output.returncode != 0
        time.sleep(self._sleep)
        if self._should_fail_test():
            if self._failed:
                return self._failure_test()
            else:
                return self._regular_test()
        else:
            return False
DEFAULT_CTRL_TESTS = [CattTest('cast h264 1920x1080 / aac content from dailymotion', ['cast', 'http://www.dailymotion.com/video/x6fotne'], substring=True, check_data=('content_id', '/389149466_mp4_h264_aac_fhd.mp4')), CattTest('set volume to 50', ['volume', '50'], sleep=2, check_data=('volume_level', 0.5)), CattTest('set volume to 100', ['volume', '100'], sleep=2, check_data=('volume_level', 1.0)), CattTest('lower volume by 50 ', ['volumedown', '50'], sleep=2, check_data=('volume_level', 0.5)), CattTest('raise volume by 50', ['volumeup', '50'], sleep=2, check_data=('volume_level', 1.0)), CattTest('mute the media volume', ['volumemute', 'True'], sleep=2, check_data=('volume_muted', True)), CattTest('unmute the media volume', ['volumemute', 'False'], sleep=2, check_data=('volume_muted', False)), CattTest('cast h264 320x184 / aac content from dailymotion', ['cast', '-y', 'format=http-240-1', 'http://www.dailymotion.com/video/x6fotne'], substring=True, check_data=('content_id', '/389149466_mp4_h264_aac_ld.mp4')), CattTest('cast h264 1280x720 / aac content from youtube using default controller', ['cast', '-f', 'https://www.youtube.com/watch?v=7fhBiXjSNQc'], check_data=('status_text', 'Casting: Dj Money J   Old School Scratch mix')), CattTest('cast first audio track from audiomack album using default controller', ['cast', 'https://audiomack.com/album/phonyppl/moza-ik'], check_data=('status_text', "Casting: mō'zā-ik. - Way Too Far.")), CattTest('cast h264 1280x720 / aac content directly from google commondatastorage', ['cast', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'], check_data=('content_id', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4')), CattTest('seek to 6:33', ['seek', '6:33'], sleep=2, time_test=True, check_data=('current_time', '393')), CattTest('rewind by 30 seconds', ['rewind', '30'], sleep=2, time_test=True, check_data=('current_time', '363')), CattTest('cast h264 1280x720 / aac content directly from google commondatastorage, start at 1:01', ['cast', '-t', '1:01', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'], sleep=5, time_test=True, check_data=('current_time', '61')), CattTest('try to use add cmd with default controller', ['add', 'https://www.youtube.com/watch?v=QcJoW9Lwzs0'], sleep=3, should_fail=True, check_err='This action is not supported by the default controller'), CattTest('try to use clear cmd with default controller', ['clear'], sleep=3, should_fail=True, check_err='This action is not supported by the default controller')]
YOUTUBE_CTRL_TESTS = [CattTest('cast video from youtube', ['cast', 'https://www.youtube.com/watch?v=mwPSIb3kt_4'], check_data=('content_id', 'mwPSIb3kt_4')), CattTest('cast video from youtube, start at 2:02', ['cast', '-t', '2:02', 'https://www.youtube.com/watch?v=mwPSIb3kt_4'], sleep=5, time_test=True, check_data=('current_time', '122')), CattTest('cast playlist from youtube', ['cast', 'https://www.youtube.com/watch?list=PLQNHYNv9IpSzzaQMuH7ji2bEy6o8T8Wwn'], check_data=('content_id', 'CIvzV5ZdYis')), CattTest('skip to next entry in playlist', ['skip'], sleep=15, check_data=('content_id', 'Ff_FvEkuG8w')), CattTest('try to add invalid video-url to playlist', ['add', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'], sleep=3, should_fail=True, check_err='This url cannot be added to the queue')]
DASHCAST_CTRL_TESTS = [CattTest('cast GitHub website frontpage', ['cast_site', 'https://github.com'], substring=True, check_data=('status_text', 'GitHub'))]
AUDIO_ONLY_TESTS = [CattTest('cast audio-only DASH aac content from facebook', ['cast', 'https://www.facebook.com/PixarCars/videos/10158549620120183/'], substring=True, check_data=('content_id', '18106055_10158549666610183_8333687643300691968_n.mp4')), CattTest('cast audio-only DASH aac content from youtube', ['cast', 'https://www.youtube.com/watch?v=7fhBiXjSNQc'], check_data=('status_text', 'Casting: Dj Money J   Old School Scratch mix')), CattTest('cast first video from youtube playlist on default controller', ['cast', 'https://www.youtube.com/watch?v=jSL1nXza7pM&list=PLAxEbmfNXWuIhN2ppUdbvXCKwalXYvs8V&index=2&t=0s'], check_data=('status_text', 'Casting: DAF - Liebe auf den Ersten Blick')), CattTest('cast "http" format audio content from mixcloud (testing format hack)', ['cast', 'https://www.mixcloud.com/Jazzmo/in-the-zone-march-2019-guidos-lounge-cafe/'], substring=True, check_data=('content_id', '/c/m4a/64/b/2/c/2/0d0c-d480-4c6a-9a9f-f485bd73bc45.m4a?sig=d65siY8itREY5iOVdGwC8w')), CattTest('cast "wav" format audio content from bandcamp (testing format hack)', ['cast', 'https://physicallysick.bandcamp.com/track/freak-is-out'], substring=True, check_data=('content_id', 'track?enc=flac'))]
STANDARD_TESTS = DEFAULT_CTRL_TESTS + YOUTUBE_CTRL_TESTS + DASHCAST_CTRL_TESTS
AUDIO_TESTS = AUDIO_ONLY_TESTS
ULTRA_TESTS = []

def run_tests(standard='', audio='', ultra=''):
    if not standard and (not audio) and (not ultra):
        raise CattTestError('No test devices were specified.')
    test_outcomes = []
    all_suites = zip([standard, audio, ultra], [STANDARD_TESTS, AUDIO_TESTS, ULTRA_TESTS])
    suites_to_run = {}
    scan_result = json.loads(subp_run(SCAN_CMD).stdout)
    for (device_name, suite) in all_suites:
        if not device_name:
            continue
        if device_name not in scan_result.keys():
            raise CattTestError('Specified device "{}" not found.'.format(device_name))
        suites_to_run.update({device_name: suite})
    for (device_name, suite) in suites_to_run.items():
        click.secho('Running some tests on "{}".'.format(device_name), fg='yellow')
        click.secho('------------------------------------------', fg='yellow')
        cbase = CMD_BASE + [device_name]
        for test in suite:
            test.set_cmd_base(cbase)
            click.echo(test.desc + '  ->  ', nl=False)
            if test.run():
                click.secho('test success!', fg='green')
                test_outcomes.append(True)
            else:
                click.secho('test failure!', fg='red')
                test_outcomes.append(False)
            if test.dump:
                click.echo('\n' + test.dump + '\n')
        subp_run(cbase + STOP_ARGS)
    return all((t for t in test_outcomes)) if test_outcomes else False

@click.command()
@click.option('-s', '--standard', help='Name of standard chromecast device.')
@click.option('-a', '--audio', help='Name of audio chromecast device.')
@click.option('-u', '--ultra', help='Name of ultra chromecast device.')
def cli(standard, audio, ultra):
    if run_tests(standard=standard, audio=audio, ultra=ultra):
        click.echo('\nAll tests were successfully completed.')
    else:
        raise CattTestError('Some tests were not successful.')
if __name__ == '__main__':
    cli()

# File: realcc_tests\test_procedure_gpt4o.py

import json
import subprocess
import time
from typing import Any, List, Tuple, Dict, Optional
import click
CMD_BASE = ['catt', '-d']
VALIDATE_ARGS = ['info', '-j']
STOP_ARGS = ['stop']
SCAN_CMD = ['catt', 'scan', '-j']

def subp_run(cmd, allow_failure=False):
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if not allow_failure and output.returncode != 0:
        raise CattTestError('The command "{}" failed.'.format(' '.join(cmd)))
    return output

class CattTestError(click.ClickException):
    pass

class CattTest:

    def __init__(self, desc, cmd_args, sleep=10, should_fail=False, substring=False, time_test=False, check_data=None, check_err=''):
        if should_fail and (not check_err) or (not should_fail and (not check_data)):
            raise CattTestError('Expected outcome mismatch.')
        if substring and time_test:
            raise CattTestError('Test type mismatch.')
        self._cmd_args = cmd_args
        self._cmd = []
        self._validate_cmd = []
        self._sleep = sleep
        self._should_fail = should_fail
        self._substring = substring
        self._time_test = time_test
        self._check_key = check_data[0] if check_data else None
        self._check_val = check_data[1] if check_data else None
        self._check_err = check_err
        self._output = None
        self._failed = False
        self.desc = desc + (' (should fail)' if self._should_fail else '')
        self.dump = ''

    def set_cmd_base(self, base):
        self._cmd = base + self._cmd_args
        self._validate_cmd = base + VALIDATE_ARGS

    def _get_val(self, key):
        output = subp_run(self._validate_cmd)
        catt_json = json.loads(output.stdout)
        return catt_json[key]

    def _should_fail_test(self):
        if self._should_fail == self._failed:
            return True
        else:
            self.dump += self._output.stderr if self._failed else self._output.stdout
            return False

    def _failure_test(self):
        output_errmsg = self._output.stderr.splitlines()[-1]
        if output_errmsg == 'Error: {}.'.format(self._check_err):
            self.dump += '{}\n - The expected error message.'.format(output_errmsg)
            return True
        else:
            self.dump += self._output.stderr
            return False

    def _regular_test(self, time_margin=5):
        catt_val = self._get_val(self._check_key)
        if self._time_test:
            passed = abs(int(catt_val) - int(self._check_val)) <= time_margin
            extra_info = '(time margin is {} seconds)'.format(time_margin)
        elif self._substring:
            passed = self._check_val in catt_val
            extra_info = '(substring)'
        else:
            passed = catt_val == self._check_val
            extra_info = ''
        if not passed:
            self.dump += 'Expected data from "{}" key:\n{} {}\nActual data:\n{}'.format(self._check_key, self._check_val, extra_info, catt_val)
        return passed

    def run(self):
        self._output = subp_run(self._cmd, allow_failure=True)
        self._failed = self._output.returncode != 0
        time.sleep(self._sleep)
        if self._should_fail_test():
            if self._failed:
                return self._failure_test()
            else:
                return self._regular_test()
        else:
            return False
DEFAULT_CTRL_TESTS = [CattTest('cast h264 1920x1080 / aac content from dailymotion', ['cast', 'http://www.dailymotion.com/video/x6fotne'], substring=True, check_data=('content_id', '/389149466_mp4_h264_aac_fhd.mp4')), CattTest('set volume to 50', ['volume', '50'], sleep=2, check_data=('volume_level', 0.5)), CattTest('set volume to 100', ['volume', '100'], sleep=2, check_data=('volume_level', 1.0)), CattTest('lower volume by 50 ', ['volumedown', '50'], sleep=2, check_data=('volume_level', 0.5)), CattTest('raise volume by 50', ['volumeup', '50'], sleep=2, check_data=('volume_level', 1.0)), CattTest('mute the media volume', ['volumemute', 'True'], sleep=2, check_data=('volume_muted', True)), CattTest('unmute the media volume', ['volumemute', 'False'], sleep=2, check_data=('volume_muted', False)), CattTest('cast h264 320x184 / aac content from dailymotion', ['cast', '-y', 'format=http-240-1', 'http://www.dailymotion.com/video/x6fotne'], substring=True, check_data=('content_id', '/389149466_mp4_h264_aac_ld.mp4')), CattTest('cast h264 1280x720 / aac content from youtube using default controller', ['cast', '-f', 'https://www.youtube.com/watch?v=7fhBiXjSNQc'], check_data=('status_text', 'Casting: Dj Money J   Old School Scratch mix')), CattTest('cast first audio track from audiomack album using default controller', ['cast', 'https://audiomack.com/album/phonyppl/moza-ik'], check_data=('status_text', "Casting: mō'zā-ik. - Way Too Far.")), CattTest('cast h264 1280x720 / aac content directly from google commondatastorage', ['cast', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'], check_data=('content_id', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4')), CattTest('seek to 6:33', ['seek', '6:33'], sleep=2, time_test=True, check_data=('current_time', '393')), CattTest('rewind by 30 seconds', ['rewind', '30'], sleep=2, time_test=True, check_data=('current_time', '363')), CattTest('cast h264 1280x720 / aac content directly from google commondatastorage, start at 1:01', ['cast', '-t', '1:01', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'], sleep=5, time_test=True, check_data=('current_time', '61')), CattTest('try to use add cmd with default controller', ['add', 'https://www.youtube.com/watch?v=QcJoW9Lwzs0'], sleep=3, should_fail=True, check_err='This action is not supported by the default controller'), CattTest('try to use clear cmd with default controller', ['clear'], sleep=3, should_fail=True, check_err='This action is not supported by the default controller')]
YOUTUBE_CTRL_TESTS = [CattTest('cast video from youtube', ['cast', 'https://www.youtube.com/watch?v=mwPSIb3kt_4'], check_data=('content_id', 'mwPSIb3kt_4')), CattTest('cast video from youtube, start at 2:02', ['cast', '-t', '2:02', 'https://www.youtube.com/watch?v=mwPSIb3kt_4'], sleep=5, time_test=True, check_data=('current_time', '122')), CattTest('cast playlist from youtube', ['cast', 'https://www.youtube.com/watch?list=PLQNHYNv9IpSzzaQMuH7ji2bEy6o8T8Wwn'], check_data=('content_id', 'CIvzV5ZdYis')), CattTest('skip to next entry in playlist', ['skip'], sleep=15, check_data=('content_id', 'Ff_FvEkuG8w')), CattTest('try to add invalid video-url to playlist', ['add', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'], sleep=3, should_fail=True, check_err='This url cannot be added to the queue')]
DASHCAST_CTRL_TESTS = [CattTest('cast GitHub website frontpage', ['cast_site', 'https://github.com'], substring=True, check_data=('status_text', 'GitHub'))]
AUDIO_ONLY_TESTS = [CattTest('cast audio-only DASH aac content from facebook', ['cast', 'https://www.facebook.com/PixarCars/videos/10158549620120183/'], substring=True, check_data=('content_id', '18106055_10158549666610183_8333687643300691968_n.mp4')), CattTest('cast audio-only DASH aac content from youtube', ['cast', 'https://www.youtube.com/watch?v=7fhBiXjSNQc'], check_data=('status_text', 'Casting: Dj Money J   Old School Scratch mix')), CattTest('cast first video from youtube playlist on default controller', ['cast', 'https://www.youtube.com/watch?v=jSL1nXza7pM&list=PLAxEbmfNXWuIhN2ppUdbvXCKwalXYvs8V&index=2&t=0s'], check_data=('status_text', 'Casting: DAF - Liebe auf den Ersten Blick')), CattTest('cast "http" format audio content from mixcloud (testing format hack)', ['cast', 'https://www.mixcloud.com/Jazzmo/in-the-zone-march-2019-guidos-lounge-cafe/'], substring=True, check_data=('content_id', '/c/m4a/64/b/2/c/2/0d0c-d480-4c6a-9a9f-f485bd73bc45.m4a?sig=d65siY8itREY5iOVdGwC8w')), CattTest('cast "wav" format audio content from bandcamp (testing format hack)', ['cast', 'https://physicallysick.bandcamp.com/track/freak-is-out'], substring=True, check_data=('content_id', 'track?enc=flac'))]
STANDARD_TESTS = DEFAULT_CTRL_TESTS + YOUTUBE_CTRL_TESTS + DASHCAST_CTRL_TESTS
AUDIO_TESTS = AUDIO_ONLY_TESTS
ULTRA_TESTS = []

def run_tests(standard='', audio='', ultra=''):
    if not standard and (not audio) and (not ultra):
        raise CattTestError('No test devices were specified.')
    test_outcomes = []
    all_suites = zip([standard, audio, ultra], [STANDARD_TESTS, AUDIO_TESTS, ULTRA_TESTS])
    suites_to_run = {}
    scan_result = json.loads(subp_run(SCAN_CMD).stdout)
    for (device_name, suite) in all_suites:
        if not device_name:
            continue
        if device_name not in scan_result.keys():
            raise CattTestError('Specified device "{}" not found.'.format(device_name))
        suites_to_run.update({device_name: suite})
    for (device_name, suite) in suites_to_run.items():
        click.secho('Running some tests on "{}".'.format(device_name), fg='yellow')
        click.secho('------------------------------------------', fg='yellow')
        cbase = CMD_BASE + [device_name]
        for test in suite:
            test.set_cmd_base(cbase)
            click.echo(test.desc + '  ->  ', nl=False)
            if test.run():
                click.secho('test success!', fg='green')
                test_outcomes.append(True)
            else:
                click.secho('test failure!', fg='red')
                test_outcomes.append(False)
            if test.dump:
                click.echo('\n' + test.dump + '\n')
        subp_run(cbase + STOP_ARGS)
    return all((t for t in test_outcomes)) if test_outcomes else False

@click.command()
@click.option('-s', '--standard', help='Name of standard chromecast device.')
@click.option('-a', '--audio', help='Name of audio chromecast device.')
@click.option('-u', '--ultra', help='Name of ultra chromecast device.')
def cli(standard, audio, ultra):
    if run_tests(standard=standard, audio=audio, ultra=ultra):
        click.echo('\nAll tests were successfully completed.')
    else:
        raise CattTestError('Some tests were not successful.')
if __name__ == '__main__':
    cli()

# File: tests\__init__.py



# File: tests\__init___gpt4o.py

def add(a, b):
    return a + b

def greet(name):
    return f'Hello, {name}!'

def is_even(number):
    return number % 2 == 0

def get_length(items):
    return len(items)

def find_max(numbers):
    return max(numbers)

# File: tests\test_catt.py

import unittest
from yt_dlp.utils import DownloadError
from catt.stream_info import StreamInfo

def ignore_tmr_failure(func):
    """
    Ignore "Too many requests" failures in a test.

    YouTube will sometimes throttle us and cause the tests to flap. This decorator
    catches the "Too many requests" exceptions in tests and ignores them.
    """

    def wrapper(*args):
        try:
            return func(*args)
        except DownloadError as err:
            if 'HTTP Error 429:' in str(err):
                pass
            else:
                raise
    return wrapper

class TestThings(unittest.TestCase):

    @ignore_tmr_failure
    def test_stream_info_youtube_video(self):
        stream = StreamInfo('https://www.youtube.com/watch?v=VZMfhtKa-wo', throw_ytdl_dl_errs=True)
        self.assertIn('https://', stream.video_url)
        self.assertEqual(stream.video_id, 'VZMfhtKa-wo')
        self.assertTrue(stream.is_remote_file)
        self.assertEqual(stream.extractor, 'youtube')

    @ignore_tmr_failure
    def test_stream_info_youtube_playlist(self):
        stream = StreamInfo('https://www.youtube.com/playlist?list=PL9Z0stL3aRykWNoVQW96JFIkelka_93Sc', throw_ytdl_dl_errs=True)
        self.assertIsNone(stream.video_url)
        self.assertEqual(stream.playlist_id, 'PL9Z0stL3aRykWNoVQW96JFIkelka_93Sc')
        self.assertTrue(stream.is_playlist)
        self.assertEqual(stream.extractor, 'youtube')

    def test_stream_info_other_video(self):
        stream = StreamInfo('https://www.twitch.tv/twitch/clip/MistySoftPenguinKappaPride')
        self.assertIn('https://', stream.video_url)
        self.assertEqual(stream.video_id, '492743767')
        self.assertTrue(stream.is_remote_file)
        self.assertEqual(stream.extractor, 'twitch')
if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())