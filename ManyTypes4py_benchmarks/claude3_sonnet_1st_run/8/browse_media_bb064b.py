"""Support for media browsing."""
import json
from typing import Any, Dict, List, Optional, Union

from homeassistant.components.media_player import BrowseError, BrowseMedia, MediaClass, MediaType

PLAYABLE_ITEM_TYPES: List[str] = ['folder', 'song', 'mywebradio', 'webradio', 'playlist', 'cuesong', 'remdisk', 'cuefile', 'folder-with-favourites', 'internal-folder']
NON_EXPANDABLE_ITEM_TYPES: List[str] = ['song', 'webradio', 'mywebradio', 'cuesong', 'album', 'artist', 'cd', 'play-playlist']
PLAYLISTS_URI_PREFIX: str = 'playlists'
ARTISTS_URI_PREFIX: str = 'artists://'
ALBUMS_URI_PREFIX: str = 'albums://'
GENRES_URI_PREFIX: str = 'genres://'
RADIO_URI_PREFIX: str = 'radio'
LAST_100_URI_PREFIX: str = 'Last_100'
FAVOURITES_URI: str = 'favourites'

def _item_to_children_media_class(item: Dict[str, Any], info: Optional[Dict[str, Any]] = None) -> MediaClass:
    if info and 'album' in info and ('artist' in info):
        return MediaClass.TRACK
    if item['uri'].startswith(PLAYLISTS_URI_PREFIX):
        return MediaClass.PLAYLIST
    if item['uri'].startswith(ARTISTS_URI_PREFIX):
        if len(item['uri']) > len(ARTISTS_URI_PREFIX):
            return MediaClass.ALBUM
        return MediaClass.ARTIST
    if item['uri'].startswith(ALBUMS_URI_PREFIX):
        if len(item['uri']) > len(ALBUMS_URI_PREFIX):
            return MediaClass.TRACK
        return MediaClass.ALBUM
    if item['uri'].startswith(GENRES_URI_PREFIX):
        if len(item['uri']) > len(GENRES_URI_PREFIX):
            return MediaClass.ALBUM
        return MediaClass.GENRE
    if item['uri'].startswith(LAST_100_URI_PREFIX) or item['uri'] == FAVOURITES_URI:
        return MediaClass.TRACK
    if item['uri'].startswith(RADIO_URI_PREFIX):
        return MediaClass.CHANNEL
    return MediaClass.DIRECTORY

def _item_to_media_class(item: Dict[str, Any], parent_item: Optional[Dict[str, Any]] = None) -> MediaClass:
    if 'type' not in item:
        return MediaClass.DIRECTORY
    if item['type'] in ('webradio', 'mywebradio'):
        return MediaClass.CHANNEL
    if item['type'] in ('song', 'cuesong'):
        return MediaClass.TRACK
    if item.get('artist'):
        return MediaClass.ALBUM
    if item['uri'].startswith(ARTISTS_URI_PREFIX) and len(item['uri']) > len(ARTISTS_URI_PREFIX):
        return MediaClass.ARTIST
    if parent_item:
        return _item_to_children_media_class(parent_item)
    return MediaClass.DIRECTORY

def _list_payload(item: Dict[str, Any], children: Optional[List[BrowseMedia]] = None) -> BrowseMedia:
    return BrowseMedia(title=item['name'], media_class=MediaClass.DIRECTORY, children_media_class=_item_to_children_media_class(item), media_content_type=MediaType.MUSIC, media_content_id=json.dumps(item), can_play=False, can_expand=True)

def _raw_item_payload(entity: Any, item: Dict[str, Any], parent_item: Optional[Dict[str, Any]] = None, title: Optional[str] = None, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if 'type' in item:
        if (thumbnail := item.get('albumart')):
            item_hash = str(hash(thumbnail))
            entity.thumbnail_cache.setdefault(item_hash, thumbnail)
            thumbnail = entity.get_browse_image_url(MediaType.MUSIC, item_hash)
    else:
        thumbnail = None
    return {'title': title or item.get('title'), 'media_class': _item_to_media_class(item, parent_item), 'children_media_class': _item_to_children_media_class(item, info), 'media_content_type': MediaType.MUSIC, 'media_content_id': json.dumps(item), 'can_play': item.get('type') in PLAYABLE_ITEM_TYPES, 'can_expand': item.get('type') not in NON_EXPANDABLE_ITEM_TYPES, 'thumbnail': thumbnail}

def _item_payload(entity: Any, item: Dict[str, Any], parent_item: Dict[str, Any]) -> BrowseMedia:
    return BrowseMedia(**_raw_item_payload(entity, item, parent_item=parent_item))

async def browse_top_level(media_library: Any) -> BrowseMedia:
    """Browse the top-level of a Volumio media hierarchy."""
    navigation: Dict[str, Any] = await media_library.browse()
    children: List[BrowseMedia] = [_list_payload(item) for item in navigation['lists']]
    return BrowseMedia(media_class=MediaClass.DIRECTORY, media_content_id='library', media_content_type='library', title='Media Library', can_play=False, can_expand=True, children=children)

async def browse_node(entity: Any, media_library: Any, media_content_type: str, media_content_id: str) -> BrowseMedia:
    """Browse a node of a Volumio media hierarchy."""
    json_item: Dict[str, Any] = json.loads(media_content_id)
    navigation: Dict[str, Any] = await media_library.browse(json_item['uri'])
    if 'lists' not in navigation:
        raise BrowseError(f'Media not found: {media_content_type} / {media_content_id}')
    first_list: Dict[str, Any] = navigation['lists'][0]
    children: List[BrowseMedia] = [_item_payload(entity, item, parent_item=json_item) for item in first_list['items']]
    info: Optional[Dict[str, Any]] = navigation.get('info')
    if not (title := first_list.get('title')):
        if info:
            title = f"{info.get('album')} ({info.get('artist')})"
        else:
            title = 'Media Library'
    payload: Dict[str, Any] = _raw_item_payload(entity, json_item, title=title, info=info)
    return BrowseMedia(**payload, children=children)
