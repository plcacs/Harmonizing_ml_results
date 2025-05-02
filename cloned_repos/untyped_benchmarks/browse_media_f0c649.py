"""Browse media for forked-daapd."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote, unquote
from homeassistant.components.media_player import BrowseError, BrowseMedia, MediaClass, MediaType
from homeassistant.helpers.network import is_internal_request
from .const import CAN_PLAY_TYPE, URI_SCHEMA
if TYPE_CHECKING:
    from . import media_player
MEDIA_TYPE_DIRECTORY = 'directory'
TOP_LEVEL_LIBRARY = {'Albums': (MediaClass.ALBUM, MediaType.ALBUM, ''), 'Artists': (MediaClass.ARTIST, MediaType.ARTIST, ''), 'Playlists': (MediaClass.PLAYLIST, MediaType.PLAYLIST, ''), 'Albums by Genre': (MediaClass.GENRE, MediaType.GENRE, MediaType.ALBUM), 'Tracks by Genre': (MediaClass.GENRE, MediaType.GENRE, MediaType.TRACK), 'Artists by Genre': (MediaClass.GENRE, MediaType.GENRE, MediaType.ARTIST), 'Directories': (MediaClass.DIRECTORY, MEDIA_TYPE_DIRECTORY, '')}
MEDIA_TYPE_TO_MEDIA_CLASS = {MediaType.ALBUM: MediaClass.ALBUM, MediaType.APP: MediaClass.APP, MediaType.ARTIST: MediaClass.ARTIST, MediaType.TRACK: MediaClass.TRACK, MediaType.PLAYLIST: MediaClass.PLAYLIST, MediaType.GENRE: MediaClass.GENRE, MEDIA_TYPE_DIRECTORY: MediaClass.DIRECTORY}
CAN_EXPAND_TYPE = {MediaType.ALBUM, MediaType.ARTIST, MediaType.PLAYLIST, MediaType.GENRE, MEDIA_TYPE_DIRECTORY}
OWNTONE_TYPE_TO_MEDIA_TYPE = {'track': MediaType.TRACK, 'playlist': MediaType.PLAYLIST, 'artist': MediaType.ARTIST, 'album': MediaType.ALBUM, 'genre': MediaType.GENRE, MediaType.APP: MediaType.APP, MEDIA_TYPE_DIRECTORY: MEDIA_TYPE_DIRECTORY}
MEDIA_TYPE_TO_OWNTONE_TYPE = {v: k for k, v in OWNTONE_TYPE_TO_MEDIA_TYPE.items()}

@dataclass
class MediaContent:
    """Class for representing OwnTone media content."""

    def __init__(self, media_content_id):
        """Create MediaContent from media_content_id."""
        _schema, self.title, _library, self.type, self.id_or_path, self.subtype = media_content_id.split(':')
        self.title = unquote(self.title)
        self.id_or_path = unquote(self.id_or_path)
        self.type = OWNTONE_TYPE_TO_MEDIA_TYPE[self.type]

def create_owntone_uri(media_type, id_or_path):
    """Create an OwnTone uri."""
    return f'library:{MEDIA_TYPE_TO_OWNTONE_TYPE[media_type]}:{quote(id_or_path)}'

def create_media_content_id(title, owntone_uri='', media_type='', id_or_path='', subtype=''):
    """Create a media_content_id.

    Either owntone_uri or both type and id_or_path must be specified.
    """
    if not owntone_uri:
        owntone_uri = create_owntone_uri(media_type, id_or_path)
    return f'{URI_SCHEMA}:{quote(title)}:{owntone_uri}:{subtype}'

def is_owntone_media_content_id(media_content_id):
    """Return whether this media_content_id is from our integration."""
    return media_content_id[:len(URI_SCHEMA)] == URI_SCHEMA

def convert_to_owntone_uri(media_content_id):
    """Convert media_content_id to OwnTone URI."""
    return ':'.join(media_content_id.split(':')[2:-1])

async def get_owntone_content(master, media_content_id):
    """Create response for the given media_content_id."""
    media_content = MediaContent(media_content_id)
    result = None
    if media_content.type == MediaType.APP:
        return base_owntone_library()
    if media_content.type == MEDIA_TYPE_DIRECTORY:
        directory_path = media_content.id_or_path
        if directory_path:
            result = await master.api.get_directory(directory=directory_path)
        else:
            result = await master.api.get_directory()
        if result is None:
            raise BrowseError(f'Media not found for {media_content.type} / {media_content_id}')
        children = []
        assert isinstance(result, dict)
        for directory in result['directories']:
            path = directory['path']
            children.append(BrowseMedia(title=path, media_class=MediaClass.DIRECTORY, media_content_id=create_media_content_id(title=path, media_type=MEDIA_TYPE_DIRECTORY, id_or_path=path), media_content_type=MEDIA_TYPE_DIRECTORY, can_play=False, can_expand=True))
        result = result['tracks']['items'] + result['playlists']['items']
        return create_browse_media_response(master, media_content, cast(list[dict[str, int | str]], result), children)
    if media_content.id_or_path == '':
        if media_content.type == MediaType.ALBUM:
            result = await master.api.get_albums()
        elif media_content.type == MediaType.ARTIST:
            result = await master.api.get_artists()
        elif media_content.type == MediaType.GENRE:
            if (result := (await master.api.get_genres())):
                for item in result:
                    item['uri'] = create_owntone_uri(MediaType.GENRE, cast(str, item['name']))
        elif media_content.type == MediaType.PLAYLIST:
            result = await master.api.get_playlists()
        if result is None:
            raise BrowseError(f'Media not found for {media_content.type} / {media_content_id}')
        return create_browse_media_response(master, media_content, cast(list[dict[str, int | str]], result))
    if media_content.type == MediaType.ALBUM:
        result = await master.api.get_tracks(album_id=media_content.id_or_path)
    elif media_content.type == MediaType.ARTIST:
        result = await master.api.get_albums(artist_id=media_content.id_or_path)
    elif media_content.type == MediaType.GENRE:
        if media_content.subtype in {MediaType.ALBUM, MediaType.ARTIST, MediaType.TRACK}:
            result = await master.api.get_genre(media_content.id_or_path, media_type=media_content.subtype)
    elif media_content.type == MediaType.PLAYLIST:
        result = await master.api.get_tracks(playlist_id=media_content.id_or_path)
    if result is None:
        raise BrowseError(f'Media not found for {media_content.type} / {media_content_id}')
    return create_browse_media_response(master, media_content, cast(list[dict[str, int | str]], result))

def create_browse_media_response(master, media_content, result, children=None):
    """Convert the results into a browse media response."""
    internal_request = is_internal_request(master.hass)
    if not children:
        children = []
    for item in result:
        if item.get('data_kind') == 'spotify' or ('path' in item and cast(str, item['path']).startswith('spotify')):
            continue
        assert isinstance(item['uri'], str)
        media_type = OWNTONE_TYPE_TO_MEDIA_TYPE[item['uri'].split(':')[1]]
        title = item.get('name') or item.get('title')
        assert isinstance(title, str)
        media_content_id = create_media_content_id(title=f'{media_content.title} / {title}', owntone_uri=item['uri'], subtype=media_content.subtype)
        if (artwork := item.get('artwork_url')):
            thumbnail = master.api.full_url(cast(str, artwork)) if internal_request else master.get_browse_image_url(media_type, media_content_id)
        else:
            thumbnail = None
        children.append(BrowseMedia(title=title, media_class=MEDIA_TYPE_TO_MEDIA_CLASS[media_type], media_content_id=media_content_id, media_content_type=media_type, can_play=media_type in CAN_PLAY_TYPE, can_expand=media_type in CAN_EXPAND_TYPE, thumbnail=thumbnail))
    return BrowseMedia(title=media_content.id_or_path if media_content.type == MEDIA_TYPE_DIRECTORY else media_content.title, media_class=MEDIA_TYPE_TO_MEDIA_CLASS[media_content.type], media_content_id='', media_content_type=media_content.type, can_play=media_content.type in CAN_PLAY_TYPE, can_expand=media_content.type in CAN_EXPAND_TYPE, children=children)

def base_owntone_library():
    """Return the base of our OwnTone library."""
    children = [BrowseMedia(title=name, media_class=media_class, media_content_id=create_media_content_id(title=name, media_type=media_type, subtype=media_subtype), media_content_type=MEDIA_TYPE_DIRECTORY, can_play=False, can_expand=True) for name, (media_class, media_type, media_subtype) in TOP_LEVEL_LIBRARY.items()]
    return BrowseMedia(title='OwnTone Library', media_class=MediaClass.APP, media_content_id=create_media_content_id(title='OwnTone Library', media_type=MediaType.APP), media_content_type=MediaType.APP, can_play=False, can_expand=True, children=children, thumbnail='https://brands.home-assistant.io/_/forked_daapd/logo.png')

def library(other):
    """Create response to describe contents of library."""
    top_level_items = [BrowseMedia(title='OwnTone Library', media_class=MediaClass.APP, media_content_id=create_media_content_id(title='OwnTone Library', media_type=MediaType.APP), media_content_type=MediaType.APP, can_play=False, can_expand=True, thumbnail='https://brands.home-assistant.io/_/forked_daapd/logo.png')]
    if other:
        top_level_items.extend(other)
    return BrowseMedia(title='OwnTone', media_class=MediaClass.DIRECTORY, media_content_id='', media_content_type=MEDIA_TYPE_DIRECTORY, can_play=False, can_expand=True, children=top_level_items)