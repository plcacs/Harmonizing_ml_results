import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yt_dlp
from .error import ExtractionError
from .error import FormatError
from .error import PlaylistError
from .util import get_local_ip
from .util import guess_mime

AUDIO_DEVICE_TYPES: List[str] = ['audio', 'group']
ULTRA_MODELS: List[Tuple[str, str]] = [('Xiaomi', 'MIBOX3'), ('Unknown manufacturer', 'Chromecast Ultra')]
BEST_MAX_2K: str = 'best[width <=? 1920][height <=? 1080]'
BEST_MAX_4K: str = 'best[width <=? 3840][height <=? 2160]'
BEST_ONLY_AUDIO: str = 'bestaudio'
BEST_FALLBACK: str = '/best'
MAX_50FPS: str = '[fps <=? 50]'
TWITCH_NO_60FPS: str = '[format_id != 1080p60__source_][format_id != 720p60]'
MIXCLOUD_NO_DASH_HLS: str = '[format_id != dash-a1-x3][format_id !*= hls-6]'
BANDCAMP_NO_AIFF_ALAC: str = '[format_id != aiff-lossless][format_id != alac]'
AUDIO_FORMAT: str = BEST_ONLY_AUDIO + MIXCLOUD_NO_DASH_HLS + BANDCAMP_NO_AIFF_ALAC + BEST_FALLBACK
ULTRA_FORMAT: str = BEST_MAX_4K + BANDCAMP_NO_AIFF_ALAC
STANDARD_FORMAT: str = BEST_MAX_2K + MAX_50FPS + TWITCH_NO_60FPS + BANDCAMP_NO_AIFF_ALAC
DEFAULT_YTDL_OPTS: Dict[str, bool] = {'quiet': True, 'no_warnings': True}

class StreamInfo:

    def __init__(self, video_url: str, cast_info: Optional[Any] = None, ytdl_options: Optional[Dict[str, Any]] = None, throw_ytdl_dl_errs: bool = False) -> None:
        self._throw_ytdl_dl_errs: bool = throw_ytdl_dl_errs
        self.local_ip: Optional[str] = get_local_ip(cast_info.host) if cast_info else None
        self.port: Optional[int] = random.randrange(45000, 47000) if cast_info else None
        if '://' in video_url:
            self._ydl: yt_dlp.YoutubeDL = yt_dlp.YoutubeDL(dict(ytdl_options) if ytdl_options else DEFAULT_YTDL_OPTS)
            self._preinfo: Dict[str, Any] = self._get_stream_preinfo(video_url)
            if self._preinfo.get('ie_key'):
                self._preinfo = self._get_stream_preinfo(self._preinfo['url'])
            self.is_local_file: bool = False
            model: Optional[Tuple[str, str]] = (cast_info.manufacturer, cast_info.model_name) if cast_info else None
            cast_type: Optional[str] = cast_info.cast_type if cast_info else None
            if 'format' in self._ydl.params:
                self._best_format: str = self._ydl.params.pop('format')
            elif cast_type and cast_type in AUDIO_DEVICE_TYPES:
                self._best_format = AUDIO_FORMAT
            elif model and model in ULTRA_MODELS:
                self._best_format = ULTRA_FORMAT
            else:
                self._best_format = STANDARD_FORMAT
            if self.is_playlist:
                self._entries: List[Dict[str, Any]] = list(self._preinfo['entries'])
                self._ydl.params.update({'noplaylist': True})
                vpreinfo: Dict[str, Any] = self._get_stream_preinfo(video_url)
                self._info: Optional[Dict[str, Any]] = self._get_stream_info(vpreinfo) if 'entries' not in vpreinfo else None
            else:
                self._info = self._get_stream_info(self._preinfo)
        else:
            self._local_file: str = video_url
            self.is_local_file = True

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
        return self.is_playlist and self._info is not None

    @property
    def extractor(self) -> Optional[str]:
        return self._preinfo['extractor'].split(':')[0] if not self.is_local_file else None

    @property
    def video_title(self) -> Optional[str]:
        if self.is_local_file:
            return Path(self._local_file).stem
        elif self._is_direct_link:
            return Path(self._preinfo['webpage_url_basename']).stem
        elif self.is_remote_file or self.is_playlist_with_active_entry:
            return self._info['title']
        else:
            return None

    @property
    def video_url(self) -> Optional[str]:
        if self.is_local_file:
            return f'http://{self.local_ip}:{self.port}/?loaded_from_catt'
        elif self.is_remote_file or self.is_playlist_with_active_entry:
            return self._get_stream_url(self._info)
        else:
            return None

    @property
    def video_id(self) -> Optional[str]:
        return self._info['id'] if self.is_remote_file or self.is_playlist_with_active_entry else None

    @property
    def video_thumbnail(self) -> Optional[str]:
        return self._info.get('thumbnail') if self.is_remote_file or self.is_playlist_with_active_entry else None

    @property
    def guessed_content_type(self) -> Optional[str]:
        if self.is_local_file:
            return guess_mime(Path(self._local_file).name)
        elif self._is_direct_link:
            return guess_mime(self._info['webpage_url_basename'])
        else:
            return None

    @property
    def guessed_content_category(self) -> Optional[str]:
        content_type = self.guessed_content_type
        return content_type.split('/')[0] if content_type else None

    @property
    def playlist_length(self) -> Optional[int]:
        return len(self._entries) if self.is_playlist else None

    @property
    def playlist_all_ids(self) -> Optional[List[str]]:
        if self.is_playlist and self._entries and self._entries[0].get('id'):
            return [entry['id'] for entry in self._entries]
        else:
            return None

    @property
    def playlist_title(self) -> Optional[str]:
        return self._preinfo['title'] if self.is_playlist else None

    @property
    def playlist_id(self) -> Optional[str]:
        return self._preinfo['id'] if self.is_playlist else None

    def set_playlist_entry(self, number: int) -> None:
        if self.is_playlist:
            if self._entries[number].get('ie_key'):
                entry = self._get_stream_preinfo(self._entries[number]['url'])
            else:
                entry = self._entries[number]
            self._info = self._get_stream_info(entry)
        else:
            raise PlaylistError('Called on non-playlist')

    def _get_stream_preinfo(self, video_url: str) -> Dict[str, Any]:
        try:
            return self._ydl.extract_info(video_url, process=False)
        except yt_dlp.utils.DownloadError:
            if self._throw_ytdl_dl_errs:
                raise
            else:
                raise ExtractionError('Remote resource not found')

    def _get_stream_info(self, preinfo: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._ydl.process_ie_result(preinfo, download=False)
        except (yt_dlp.utils.ExtractorError, yt_dlp.utils.DownloadError):
            raise ExtractionError('yt-dlp extractor failed')

    def _get_stream_url(self, info: Dict[str, Any]) -> str:
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
