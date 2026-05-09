from homeassistant.components.media_player import BrowseError, BrowseMedia, MediaClass, MediaType
from typing import Optional, List

PLAYABLE_ITEM_TYPES = ['folder', 'song', 'mywebradio', 'webradio', 'playlist', 'cuesong', 'remdisk', 'cuefile', 'folder-with-favourites', 'internal-folder']
NON_EXPANDABLE_ITEM_TYPES = ['song', 'webradio', 'mywebradio', 'cuesong', 'album', 'artist', 'cd', 'play-playlist']
PLAYLISTS_URI_PREFIX = 'playlists'
ARTISTS_URI_PREFIX = 'artists://'
ALBUMS_URI_PREFIX = 'albums://'
GENRES_URI_PREFIX = 'genres://'
RADIO_URI_PREFIX = 'radio'
LAST_100_URI_PREFIX = 'Last_100'
FAVOURITES_URI = 'favourites'

def _item_to_children_media_class(item: dict, info: Optional[dict] = None) -> MediaClass:
    ...

def _item_to_media_class(item: dict, parent_item: Optional[dict] = None) -> MediaClass:
    ...

def _list_payload(item: dict, children: Optional[List[BrowseMedia]] = None) -> BrowseMedia:
    ...

def _raw_item_payload(entity, item: dict, parent_item: Optional[dict] = None, title: Optional[str] = None, info: Optional[dict] = None) -> dict:
    ...

def _item_payload(entity, item: dict, parent_item: dict) -> BrowseMedia:
    ...

async def browse_top_level(media_library: object) -> BrowseMedia:
    ...

async def browse_node(entity: object, media_library: object, media_content_type: str, media_content_id: str) -> BrowseMedia:
    ...
