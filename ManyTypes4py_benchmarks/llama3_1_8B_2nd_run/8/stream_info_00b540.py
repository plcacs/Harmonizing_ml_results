import random
from pathlib import Path
import yt_dlp
from .error import ExtractionError
from .error import FormatError
from .error import PlaylistError
from .util import get_local_ip
from .util import guess_mime

AUDIO_DEVICE_TYPES = list[str]
ULTRA_MODELS = list[tuple[str, str]]
BEST_MAX_2K = str
BEST_MAX_4K = str
BEST_ONLY_AUDIO = str
BEST_FALLBACK = str
MAX_50FPS = str
TWITCH_NO_60FPS = str
MIXCLOUD_NO_DASH_HLS = str
BANDCAMP_NO_AIFF_ALAC = str
AUDIO_FORMAT = str
ULTRA_FORMAT = str
STANDARD_FORMAT = str
DEFAULT_YTDL_OPTS = dict[str, bool]

class StreamInfo:
    def __init__(self, video_url: str, cast_info: dict[str, str] | None = None, ytdl_options: dict[str, bool] | None = None, throw_ytdl_dl_errs: bool = False) -> None:
        self._throw_ytdl_dl_errs: bool = throw_ytdl_dl_errs
        self.local_ip: str | None = get_local_ip(cast_info.get('host')) if cast_info else None
        self.port: int | None = random.randrange(45000, 47000) if cast_info else None
        if '://' in video_url:
            self._ydl: yt_dlp.YoutubeDL = yt_dlp.YoutubeDL(dict(ytdl_options) if ytdl_options else DEFAULT_YTDL_OPTS)
            self._preinfo: dict[str, str] = self._get_stream_preinfo(video_url)
            if self._preinfo.get('ie_key'):
                self._preinfo = self._get_stream_preinfo(self._preinfo['url'])
            self.is_local_file: bool = False
            model: tuple[str, str] | None = (cast_info.get('manufacturer'), cast_info.get('model_name')) if cast_info else None
            cast_type: str | None = cast_info.get('cast_type') if cast_info else None
            if 'format' in self._ydl.params:
                self._best_format: str = self._ydl.params.pop('format')
            elif cast_type and cast_type in AUDIO_DEVICE_TYPES:
                self._best_format = AUDIO_FORMAT
            elif model and model in ULTRA_MODELS:
                self._best_format = ULTRA_FORMAT
            else:
                self._best_format = STANDARD_FORMAT
            if self.is_playlist:
                self._entries: list[dict[str, str]] = list(self._preinfo['entries'])
                self._ydl.params.update({'noplaylist': True})
                vpreinfo: dict[str, str] = self._get_stream_preinfo(video_url)
                self._info: dict[str, str] | None = self._get_stream_info(vpreinfo) if 'entries' not in vpreinfo else None
            else:
                self._info: dict[str, str] = self._get_stream_info(self._preinfo)
        else:
            self._local_file: str = video_url
            self.is_local_file: bool = True

    @property
    def is_remote_file(self) -> bool:
        return not self.is_local_file and (not self.is_playlist)

    @property
    def _is_direct_link(self) -> bool:
        return self.is_remote_file and self._info.get('direct')

    @property
    def is_playlist(self) -> bool:
        return not self.is_local_file and 'entries' in self._preinfo

    @property
    def is_playlist_with_active_entry(self) -> bool:
        return self.is_playlist and self._info

    @property
    def extractor(self) -> str | None:
        return self._preinfo['extractor'].split(':')[0] if not self.is_local_file else None

    @property
    def video_title(self) -> str | None:
        if self.is_local_file:
            return Path(self._local_file).stem
        elif self._is_direct_link:
            return Path(self._preinfo['webpage_url_basename']).stem
        elif self.is_remote_file or self.is_playlist_with_active_entry:
            return self._info['title']
        else:
            return None

    @property
    def video_url(self) -> str | None:
        if self.is_local_file:
            return 'http://{}:{}/?loaded_from_catt'.format(self.local_ip, self.port)
        elif self.is_remote_file or self.is_playlist_with_active_entry:
            return self._get_stream_url(self._info)
        else:
            return None

    @property
    def video_id(self) -> str | None:
        return self._info['id'] if self.is_remote_file or self.is_playlist_with_active_entry else None

    @property
    def video_thumbnail(self) -> str | None:
        return self._info.get('thumbnail') if self.is_remote_file or self.is_playlist_with_active_entry else None

    @property
    def guessed_content_type(self) -> str | None:
        if self.is_local_file:
            return guess_mime(Path(self._local_file).name)
        elif self._is_direct_link:
            return guess_mime(self._info['webpage_url_basename'])
        else:
            return None

    @property
    def guessed_content_category(self) -> str | None:
        content_type: str | None = self.guessed_content_type
        return content_type.split('/')[0] if content_type else None

    @property
    def playlist_length(self) -> int | None:
        return len(self._entries) if self.is_playlist else None

    @property
    def playlist_all_ids(self) -> list[str] | None:
        if self.is_playlist and self._entries and self._entries[0].get('id'):
            return [entry['id'] for entry in self._entries]
        else:
            return None

    @property
    def playlist_title(self) -> str | None:
        return self._preinfo['title'] if self.is_playlist else None

    @property
    def playlist_id(self) -> str | None:
        return self._preinfo['id'] if self.is_playlist else None

    def set_playlist_entry(self, number: int) -> None:
        if self.is_playlist:
            if self._entries[number].get('ie_key'):
                entry: dict[str, str] = self._get_stream_preinfo(self._entries[number]['url'])
            else:
                entry: dict[str, str] = self._entries[number]
            self._info = self._get_stream_info(entry)
        else:
            raise PlaylistError('Called on non-playlist')

    def _get_stream_preinfo(self, video_url: str) -> dict[str, str]:
        try:
            return self._ydl.extract_info(video_url, process=False)
        except yt_dlp.utils.DownloadError:
            if self._throw_ytdl_dl_errs:
                raise
            else:
                raise ExtractionError('Remote resource not found')

    def _get_stream_info(self, preinfo: dict[str, str]) -> dict[str, str]:
        try:
            return self._ydl.process_ie_result(preinfo, download=False)
        except (yt_dlp.utils.ExtractorError, yt_dlp.utils.DownloadError):
            raise ExtractionError('yt-dlp extractor failed')

    def _get_stream_url(self, info: dict[str, str]) -> str:
        try:
            format_selector = self._ydl.build_format_selector(self._best_format)
        except ValueError:
            raise FormatError('The specified format filter is invalid')
        info.setdefault('incomplete_formats', {})
        try:
            best_format: dict[str, str] = next(format_selector(info))
        except StopIteration:
            raise FormatError('No suitable format was found')
        except KeyError:
            best_format: dict[str, str] = info
        return best_format['url']
