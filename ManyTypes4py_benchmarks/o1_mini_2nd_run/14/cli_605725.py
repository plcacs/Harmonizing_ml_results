import configparser
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
    VERSION = '0.0.0u'


class CattTimeParamType(click.ParamType):
    name: str = 'catt_time'

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> int:
        try:
            tdesc: List[int] = [int(x) for x in value.split(':')]
            tlen: int = len(tdesc)
            if (tlen > 1 and any((t > 59 for t in tdesc))) or tlen > 3:
                raise ValueError
        except ValueError:
            self.fail(f'{value} is not a valid time description.', param, ctx)
        tdesc.reverse()
        return sum((tdesc[p] * 60 ** p for p in range(tlen)))


CATT_TIME: CattTimeParamType = CattTimeParamType()


class YtdlOptParamType(click.ParamType):
    name: str = 'ytdl_option'

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Tuple[str, Union[str, bool]]:
        if '=' not in value:
            self.fail(f'{value} is not a valid key/value pair.', param, ctx)
        ykey: str
        yval: str = ''
        ykey, yval = value.split('=', 1)
        yval = {'true': True, 'false': False}.get(yval.lower().strip(), yval)
        return (ykey, yval)


YTDL_OPT: YtdlOptParamType = YtdlOptParamType()


def process_url(
    ctx: click.Context,
    param: click.Parameter,
    value: str
) -> str:
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


def process_path(
    ctx: click.Context,
    param: click.Parameter,
    value: str
) -> Optional[Path]:
    path: Optional[Path] = Path(value) if value else None
    if path and (path.is_dir() or not path.parent.exists()):
        raise CliError('The specified path is invalid')
    return path


def process_subtitles(
    ctx: click.Context,
    param: click.Parameter,
    value: Optional[str]
) -> Optional[str]:
    if not value:
        return None
    pval: str = urlparse(value).path if '://' in value else value
    if not pval.lower().endswith(('.srt', '.vtt')):
        raise CliError('Invalid subtitles format, only srt and vtt are supported')
    if '://' not in value and (not Path(value).is_file()):
        raise CliError(f'Subtitles file [{value}] does not exist')
    return value


def process_device(device_desc: Optional[str], aliases: Dict[str, str]) -> Optional[str]:
    """
    Resolve real device name when value is an alias.

    :param device_desc: Can be an ip-address or a name (alias or real name).
    :type device_desc: Optional[str]
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


def create_server_thread(
    filename: str,
    address: str,
    port: int,
    content_type: str,
    single_req: bool = False
) -> Thread:
    thr: Thread = Thread(target=serve_file, args=(filename, address, port, content_type, single_req))
    thr.setDaemon(True)
    thr.start()
    return thr


CONTEXT_SETTINGS: Dict[str, Any] = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('-d', '--device', metavar='NAME_OR_IP', help='Select Chromecast device.')
@click.version_option(
    version=VERSION,
    prog_name=PROGRAM_NAME,
    message='%(prog)s v%(version)s, ' + __codename__ + '.'
)
@click.pass_context
def cli(ctx: click.Context, device: Optional[str]) -> None:
    device_from_config: Optional[str] = ctx.obj.get('options', {}).get('device')
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
def cast(
    settings: Dict[str, Any],
    video_url: str,
    subtitles: Optional[str],
    force_default: bool,
    random_play: bool,
    no_subs: bool,
    no_playlist: bool,
    ytdl_option: Tuple[Tuple[str, Union[str, bool]], ...],
    seek_to: Optional[int],
    block: bool = False
) -> None:
    controller: Optional[str] = 'default' if force_default or ytdl_option else None
    playlist_playback: bool = False
    st_thr: Optional[Thread] = None
    su_thr: Optional[Thread] = None
    subs: Optional[SubsInfo] = None
    cst: CastState
    stream: Any
    cst, stream = setup_cast(
        settings['selected_device'],
        video_url=video_url,
        prep='app',
        controller=controller,
        ytdl_options=ytdl_option
    )
    media_is_image: bool = stream.guessed_content_category == 'image'
    local_or_remote: str = 'local' if stream.is_local_file else 'remote'
    if stream.is_local_file:
        fail_if_no_ip(stream.local_ip)
        st_thr = create_server_thread(
            video_url,
            stream.local_ip,
            stream.port,
            stream.guessed_content_type,
            single_req=media_is_image
        )
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
        click.echo(f'Casting remote playlist {video_url}...')
        video_id: Optional[str] = stream.video_id or (stream.playlist_all_ids[0] if stream.playlist_all_ids else None)
        if video_id is not None:
            cst.play_playlist(stream.playlist_id, video_id=video_id)
    else:
        if not subtitles and (not no_subs) and stream.is_local_file:
            subtitles = hunt_subtitles(video_url)
        if subtitles:
            fail_if_no_ip(stream.local_ip)
            subs = SubsInfo(subtitles, stream.local_ip, stream.port + 1)
            su_thr = create_server_thread(
                subs.file,
                subs.local_ip,
                subs.port,
                'text/vtt;charset=utf-8',
                single_req=True
            )
        click.echo(f'Casting {local_or_remote} file {video_url}...')
        click.echo(
            f'{"Showing" if media_is_image else "Playing"} "{stream.video_title}" on "{cst.cc_name}"...'
        )
        if cst.info_type == 'url':
            cst.play_media_url(
                stream.video_url,
                title=stream.video_title,
                content_type=stream.guessed_content_type,
                subtitles=subs.url if subs else None,
                thumb=stream.video_thumbnail,
                current_time=seek_to
            )
        elif cst.info_type == 'id':
            cst.play_media_id(stream.video_id, current_time=seek_to)
        else:
            raise ValueError('Invalid or undefined info type')
    if stream.is_local_file or subs:
        click.echo('Serving local file(s).')
    if not media_is_image and (stream.is_local_file or block):
        if not cst.wait_for(['PLAYING'], timeout=WAIT_PLAY_TIMEOUT):
            raise CliError(f'Playback of {local_or_remote} file has failed')
        cst.wait_for(['UNKNOWN', 'IDLE'])
    elif (stream.is_local_file and media_is_image) or subs:
        while (st_thr and st_thr.is_alive()) or (su_thr and su_thr.is_alive()):
            time.sleep(1)


@cli.command('cast_site', short_help='Cast any website to a Chromecast.')
@click.argument('url', callback=process_url)
@click.pass_obj
def cast_site(settings: Dict[str, Any], url: str) -> None:
    cst: CastState = setup_cast(settings['selected_device'], controller='dashcast', action='load_url', prep='app')
    click.echo(f'Casting {url} on "{cst.cc_name}"...')
    cst.load_url(url)


@cli.command(short_help='Add a video to the queue (YouTube only).')
@click.argument('video_url', callback=process_url)
@click.option('-n', '--play-next', is_flag=True, help='Add video immediately after currently playing video.')
@click.pass_obj
def add(settings: Dict[str, Any], video_url: str, play_next: bool) -> None:
    cst: CastState
    stream: Any
    cst, stream = setup_cast(settings['selected_device'], video_url=video_url, action='add', prep='control')
    if cst.name != stream.extractor or not (stream.is_remote_file or stream.is_playlist_with_active_entry):
        raise CliError('This url cannot be added to the queue')
    click.echo(f'Adding video id "{stream.video_id}" to the queue.')
    if play_next:
        cst.add_next(stream.video_id)
    else:
        cst.add(stream.video_id)


@cli.command(short_help='Remove a video from the queue (YouTube only).')
@click.argument('video_url', callback=process_url)
@click.pass_obj
def remove(settings: Dict[str, Any], video_url: str) -> None:
    cst: CastState
    stream: Any
    cst, stream = setup_cast(settings['selected_device'], video_url=video_url, action='remove', prep='control')
    if cst.name != stream.extractor or not stream.is_remote_file:
        raise CliError('This url cannot be removed from the queue')
    click.echo(f'Removing video id "{stream.video_id}" from the queue.')
    cst.remove(stream.video_id)


@cli.command(short_help='Clear the queue (YouTube only).')
@click.pass_obj
def clear(settings: Dict[str, Any]) -> None:
    cst: CastState = setup_cast(settings['selected_device'], action='clear', prep='control')
    cst.clear()


@cli.command(short_help='Pause a video.')
@click.pass_obj
def pause(settings: Dict[str, Any]) -> None:
    cst: CastState = setup_cast(settings['selected_device'], action='pause', prep='control')
    cst.pause()


@cli.command(short_help='Resume a video after it has been paused.')
@click.pass_obj
def play(settings: Dict[str, Any]) -> None:
    cst: CastState = setup_cast(settings['selected_device'], action='play', prep='control')
    cst.play()


@cli.command('play_toggle', short_help='Toggle between playing and paused state.')
@click.pass_obj
def play_toggle(settings: Dict[str, Any]) -> None:
    cst: CastState = setup_cast(settings['selected_device'], action='play_toggle', prep='control')
    cst.play_toggle()


@cli.command(short_help='Stop playing.')
@click.option('-f', '--force', is_flag=True, help='Launch dummy chromecast app before sending stop command (for devices that do not respond to stop command under certain circumstances).')
@click.pass_obj
def stop(settings: Dict[str, Any], force: bool) -> None:
    cst: CastState = setup_cast(settings['selected_device'])
    cst.kill(force=force)


@cli.command(short_help='Rewind a video by TIME duration.')
@click.argument('timedesc', type=CATT_TIME, required=False, default='30', metavar='TIME')
@click.pass_obj
def rewind(settings: Dict[str, Any], timedesc: int) -> None:
    cst: CastState = setup_cast(settings['selected_device'], action='rewind', prep='control')
    cst.rewind(timedesc)


@cli.command(short_help='Fastforward a video by TIME duration.')
@click.argument('timedesc', type=CATT_TIME, required=False, default='30', metavar='TIME')
@click.pass_obj
def ffwd(settings: Dict[str, Any], timedesc: int) -> None:
    cst: CastState = setup_cast(settings['selected_device'], action='ffwd', prep='control')
    cst.ffwd(timedesc)


@cli.command(short_help='Seek the video to TIME position.')
@click.argument('timedesc', type=CATT_TIME, metavar='TIME')
@click.pass_obj
def seek(settings: Dict[str, Any], timedesc: int) -> None:
    cst: CastState = setup_cast(settings['selected_device'], action='seek', prep='control')
    cst.seek(timedesc)


@cli.command(short_help='Skip to end of content.')
@click.pass_obj
def skip(settings: Dict[str, Any]) -> None:
    cst: CastState = setup_cast(settings['selected_device'], action='skip', prep='control')
    cst.skip()


@cli.command(short_help='Set the volume to LVL [0-100].')
@click.argument('level', type=click.IntRange(0, 100), metavar='LVL')
@click.pass_obj
def volume(settings: Dict[str, Any], level: int) -> None:
    cst: CastState = setup_cast(settings['selected_device'])
    cst.volume(level / 100.0)


@cli.command(short_help='Turn up volume by a DELTA increment.')
@click.argument('delta', type=click.IntRange(1, 100), required=False, default=10, metavar='DELTA')
@click.pass_obj
def volumeup(settings: Dict[str, Any], delta: int) -> None:
    cst: CastState = setup_cast(settings['selected_device'])
    cst.volumeup(delta / 100.0)


@cli.command(short_help='Turn down volume by a DELTA increment.')
@click.argument('delta', type=click.IntRange(1, 100), required=False, default=10, metavar='DELTA')
@click.pass_obj
def volumedown(settings: Dict[str, Any], delta: int) -> None:
    cst: CastState = setup_cast(settings['selected_device'])
    cst.volumedown(delta / 100.0)


@cli.command(short_help='Enable or disable mute on supported devices.')
@click.argument('muted', type=click.BOOL, required=False, default=True, metavar='MUTED')
@click.pass_obj
def volumemute(settings: Dict[str, Any], muted: bool) -> None:
    cst: CastState = setup_cast(settings['selected_device'])
    cst.volumemute(muted)


@cli.command(short_help='Show some information about the currently-playing video.')
@click.pass_obj
def status(settings: Dict[str, Any]) -> None:
    cst: CastState = setup_cast(settings['selected_device'], prep='info')
    echo_status(cst.cast_info)


@cli.command(short_help='Show complete information about the currently-playing video.')
@click.option('-j', '--json-output', is_flag=True, help='Output info as json.')
@click.pass_obj
def info(settings: Dict[str, Any], json_output: bool) -> None:
    try:
        cst: CastState = setup_cast(settings['selected_device'], prep='info')
    except CastError:
        if json_output:
            info_data: Dict[str, Any] = {}
        else:
            raise
    else:
        info_data: Dict[str, Any] = cst.info
    if json_output:
        echo_json(info_data)
    else:
        for key, value in info_data.items():
            click.echo(f'{key}: {value}')


@cli.command(short_help='Scan the local network and show all Chromecasts and their IPs.')
@click.option('-j', '--json-output', is_flag=True, help='Output scan result as json.')
def scan(json_output: bool) -> None:
    if not json_output:
        click.echo('Scanning Chromecasts...')
    devices: List[Any] = get_cast_infos()
    if json_output:
        devices_dict: Dict[str, Dict[str, Any]] = {d.friendly_name: d._asdict() for d in devices}
        echo_json(devices_dict)
    else:
        if not devices:
            raise CastError('No devices found')
        for device in devices:
            click.echo(f'{device.host} - {device.friendly_name} - {device.manufacturer} {device.model_name}')


@cli.command(short_help='Save the current state of the Chromecast for later use.')
@click.argument('path', type=click.Path(writable=True), callback=process_path, required=False)
@click.pass_obj
def save(settings: Dict[str, Any], path: Optional[Path]) -> None:
    cst: CastState = setup_cast(settings['selected_device'], prep='control')
    if not cst.save_capability or cst.is_streaming_local_file:
        raise CliError('Saving state of this kind of content is not supported')
    elif cst.save_capability == 'partial':
        echo_warning('Please be advised that playlist data will not be saved')
    echo_status(cst.cast_info)
    if path and path.is_file():
        click.confirm('File already exists. Overwrite?', abort=True)
    click.echo('Saving...')
    if path:
        state: CastState = CastState(path, StateMode.ARBI)
        cc_name: str = '*'
    else:
        state = CastState(STATE_PATH, StateMode.CONF)
        cc_name = cst.cc_name
    state.set_data(cc_name, {'controller': cst.name, 'data': cst.cast_info})


@cli.command(short_help='Return Chromecast to saved state.')
@click.argument('path', type=click.Path(exists=True), callback=process_path, required=False)
@click.pass_obj
def restore(settings: Dict[str, Any], path: Optional[Path]) -> None:
    if not path and (not STATE_PATH.is_file()):
        raise CliError('Save file in config dir has not been created')
    cst: CastState = setup_cast(settings['selected_device'])
    state: CastState = CastState(path or STATE_PATH, StateMode.READ)
    try:
        data: Dict[str, Any] = state.get_data(cst.cc_name if not path else None)
    except StateFileError:
        raise CliError('The chosen file is not a valid save file')
    if not data:
        raise CliError('No save data found for this device')
    echo_status(data['data'])
    click.echo('Restoring...')
    cst = setup_cast(
        settings['selected_device'],
        prep='app',
        controller=data['controller']
    )
    cst.restore(data['data'])


@cli.command('write_config', short_help='DEPRECATED: Please use "set_default".')
def write_config() -> None:
    raise CliError('DEPRECATED: Please use "set_default"')


@cli.command('set_default', short_help='Set the selected device as default.')
@click.pass_obj
def set_default(settings: Dict[str, Any]) -> None:
    config: configparser.ConfigParser = readconfig()
    device: str = get_device_from_settings(settings)
    config['options']['device'] = device
    writeconfig(config)


@cli.command('del_default', short_help='Delete the default device.')
@click.pass_obj
def del_default(settings: Dict[str, Any]) -> None:
    config: configparser.ConfigParser = readconfig()
    if 'device' not in config['options']:
        raise CliError('No default device is set, so none deleted')
    config['options'].pop('device')
    writeconfig(config)


@cli.command('set_alias', short_help='Set an alias name for the selected device (case-insensitive).')
@click.argument('name', type=str)
@click.pass_obj
def set_alias(settings: Dict[str, Any], name: str) -> None:
    config: configparser.ConfigParser = readconfig()
    device: str = get_device_from_settings(settings)
    old_alias: Optional[str] = get_alias_from_config(config, device)
    if old_alias:
        config['aliases'].pop(old_alias)
    config['aliases'][name] = device
    writeconfig(config)


@cli.command('del_alias', short_help='Delete the alias name of the selected device.')
@click.pass_obj
def del_alias(settings: Dict[str, Any]) -> None:
    config: configparser.ConfigParser = readconfig()
    device: str = get_device_from_settings(settings)
    alias: Optional[str] = get_alias_from_config(config, device)
    if not alias:
        raise CliError(f'No alias exists for "{device}", so none deleted')
    config['aliases'].pop(alias)
    writeconfig(config)


def get_alias_from_config(config: configparser.ConfigParser, device: str) -> Optional[str]:
    try:
        return next(a for a, d in config['aliases'].items() if d == device)
    except StopIteration:
        return None


def get_device_from_settings(settings: Dict[str, Any]) -> str:
    device_desc: Optional[str] = settings['selected_device']
    if not device_desc or not settings.get('selected_device_is_from_cli', False):
        raise CliError('No device specified (must be explicitly specified with -d option)')
    is_ip: bool = is_ipaddress(device_desc)
    if is_ip:
        found: bool = cast_ip_exists(device_desc)
    else:
        found = device_desc in [d.friendly_name for d in get_cast_infos()]
    if not found:
        if is_ip:
            msg = f'No device found at {device_desc}'
        else:
            msg = f'Specified device "{device_desc}" not found'
        raise CliError(msg)
    return device_desc


def writeconfig(config: configparser.ConfigParser) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True)
    except FileExistsError:
        pass
    with CONFIG_PATH.open('w') as configfile:
        config.write(configfile)


def readconfig() -> configparser.ConfigParser:
    config: configparser.ConfigParser = configparser.ConfigParser()
    config.read(str(CONFIG_PATH))
    for req_section in ('options', 'aliases'):
        if req_section not in config.sections():
            config.add_section(req_section)
    return config


def get_config_as_dict() -> Dict[str, Dict[str, str]]:
    """
    Returns a dictionary of the form:
        {"options": {"key": "value"},
         "aliases": {"device1": "device_name"}}
    """
    config: configparser.ConfigParser = readconfig()
    return {section: dict(config.items(section)) for section in config.sections()}


def main() -> None:
    try:
        cli(obj=get_config_as_dict())
    except CattUserError as err:
        sys.exit(f'Error: {str(err)}.')


if __name__ == '__main__':
    main()
