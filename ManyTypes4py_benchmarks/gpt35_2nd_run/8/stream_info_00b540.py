from typing import Optional, Dict, List, Union

class StreamInfo:

    def __init__(self, video_url: str, cast_info: Optional[CastInfo] = None, ytdl_options: Optional[Dict] = None, throw_ytdl_dl_errs: bool = False) -> None:

    @property
    def is_remote_file(self) -> bool:

    @property
    def _is_direct_link(self) -> bool:

    @property
    def is_playlist(self) -> bool:

    @property
    def is_playlist_with_active_entry(self) -> bool:

    @property
    def extractor(self) -> Optional[str]:

    @property
    def video_title(self) -> Optional[str]:

    @property
    def video_url(self) -> Optional[str]:

    @property
    def video_id(self) -> Optional[str]:

    @property
    def video_thumbnail(self) -> Optional[str]:

    @property
    def guessed_content_type(self) -> Optional[str]:

    @property
    def guessed_content_category(self) -> Optional[str]:

    @property
    def playlist_length(self) -> Optional[int]:

    @property
    def playlist_all_ids(self) -> Optional[List[str]]:

    @property
    def playlist_title(self) -> Optional[str]:

    @property
    def playlist_id(self) -> Optional[str]:

    def set_playlist_entry(self, number: int) -> None:

    def _get_stream_preinfo(self, video_url: str) -> Dict:

    def _get_stream_info(self, preinfo: Dict) -> Dict:

    def _get_stream_url(self, info: Dict) -> str:
