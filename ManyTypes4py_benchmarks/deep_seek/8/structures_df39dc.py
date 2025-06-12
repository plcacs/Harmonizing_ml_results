import json
import lzma
import re
from base64 import b64decode, b64encode
from contextlib import suppress
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union, cast
from unicodedata import normalize
from . import __version__
from .exceptions import *
from .instaloadercontext import InstaloaderContext
from .nodeiterator import FrozenNodeIterator, NodeIterator
from .sectioniterator import SectionIterator

class PostSidecarNode(NamedTuple):
    """Item of a Sidecar Post."""
    is_video: bool
    display_url: str
    video_url: Optional[str]

PostSidecarNode.is_video.__doc__ = 'Whether this node is a video.'
PostSidecarNode.display_url.__doc__ = 'URL of image or video thumbnail.'
PostSidecarNode.video_url.__doc__ = 'URL of video or None.'

class PostCommentAnswer(NamedTuple):
    id: int
    created_at_utc: datetime
    text: str
    owner: 'Profile'
    likes_count: int

PostCommentAnswer.id.__doc__ = 'ID number of comment.'
PostCommentAnswer.created_at_utc.__doc__ = ':class:`~datetime.datetime` when comment was created (UTC).'
PostCommentAnswer.text.__doc__ = 'Comment text.'
PostCommentAnswer.owner.__doc__ = 'Owner :class:`Profile` of the comment.'
PostCommentAnswer.likes_count.__doc__ = 'Number of likes on comment.'

class PostComment:
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any], answers: Iterable[PostCommentAnswer], post: 'Post') -> None:
        self._context = context
        self._node = node
        self._answers = answers
        self._post = post

    @classmethod
    def from_iphone_struct(cls, context: InstaloaderContext, media: Dict[str, Any], answers: Iterable[PostCommentAnswer], post: 'Post') -> 'PostComment':
        return cls(context=context, node={'id': int(media['pk']), 'created_at': media['created_at'], 'text': media['text'], 'edge_liked_by': {'count': media['comment_like_count']}, 'iphone_struct': media}, answers=answers, post=post)

    @property
    def id(self) -> int:
        """ ID number of comment. """
        return self._node['id']

    @property
    def created_at_utc(self) -> datetime:
        """ :class:`~datetime.datetime` when comment was created (UTC). """
        return datetime.utcfromtimestamp(self._node['created_at'])

    @property
    def text(self) -> str:
        """ Comment text. """
        return self._node['text']

    @property
    def owner(self) -> 'Profile':
        """ Owner :class:`Profile` of the comment. """
        if 'iphone_struct' in self._node:
            return Profile.from_iphone_struct(self._context, self._node['iphone_struct']['user'])
        return Profile(self._context, self._node['owner'])

    @property
    def likes_count(self) -> int:
        """ Number of likes on comment. """
        return self._node.get('edge_liked_by', {}).get('count', 0)

    @property
    def answers(self) -> Iterable[PostCommentAnswer]:
        """ Iterator which yields all :class:`PostCommentAnswer` for the comment. """
        return self._answers

    @property
    def likes(self) -> Union[NodeIterator['Profile'], List[Any]]:
        """
        Iterate over all likes of a comment. A :class:`Profile` instance of each like is yielded.

        .. versionadded:: 4.11
        """
        if self.likes_count != 0:
            return NodeIterator(self._context, '5f0b1f6281e72053cbc07909c8d154ae', lambda d: d['data']['comment']['edge_liked_by'], lambda n: Profile(self._context, n), {'comment_id': self.id}, 'https://www.instagram.com/p/{0}/'.format(self._post.shortcode))
        return []

    def __repr__(self) -> str:
        return f'<PostComment {self.id} of {self._post.shortcode}>'

class PostLocation(NamedTuple):
    id: int
    name: str
    slug: str
    has_public_page: bool
    lat: Optional[float]
    lng: Optional[float]

PostLocation.id.__doc__ = 'ID number of location.'
PostLocation.name.__doc__ = 'Location name.'
PostLocation.slug.__doc__ = 'URL friendly variant of location name.'
PostLocation.has_public_page.__doc__ = 'Whether location has a public page.'
PostLocation.lat.__doc__ = 'Latitude (:class:`float` or None).'
PostLocation.lng.__doc__ = 'Longitude (:class:`float` or None).'

_hashtag_regex = re.compile('(?:#)((?:\\w){1,150})')
_mention_regex = re.compile('(?:^|[^\\w\\n]|_)(?:@)(\\w(?:(?:\\w|(?:\\.(?!\\.))){0,28}(?:\\w))?)', re.ASCII)

def _optional_normalize(string: Optional[str]) -> Optional[str]:
    if string is not None:
        return normalize('NFC', string)
    else:
        return None

class Post:
    """
    Structure containing information about an Instagram post.
    """
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any], owner_profile: Optional['Profile'] = None) -> None:
        assert 'shortcode' in node or 'code' in node
        self._context = context
        self._node = node
        self._owner_profile = owner_profile
        self._full_metadata_dict: Optional[Dict[str, Any]] = None
        self._location: Optional[PostLocation] = None
        self._iphone_struct_: Optional[Dict[str, Any]] = None
        if 'iphone_struct' in node:
            self._iphone_struct_ = node['iphone_struct']

    @classmethod
    def from_shortcode(cls, context: InstaloaderContext, shortcode: str) -> 'Post':
        """Create a post object from a given shortcode"""
        post = cls(context, {'shortcode': shortcode})
        post._node = post._full_metadata
        return post

    @classmethod
    def from_mediaid(cls, context: InstaloaderContext, mediaid: int) -> 'Post':
        """Create a post object from a given mediaid"""
        return cls.from_shortcode(context, Post.mediaid_to_shortcode(mediaid))

    @classmethod
    def from_iphone_struct(cls, context: InstaloaderContext, media: Dict[str, Any]) -> 'Post':
        """Create a post from a given iphone_struct."""
        media_types = {1: 'GraphImage', 2: 'GraphVideo', 8: 'GraphSidecar'}
        fake_node = {'shortcode': media['code'], 'id': media['pk'], '__typename': media_types[media['media_type']], 'is_video': media_types[media['media_type']] == 'GraphVideo', 'date': media['taken_at'], 'caption': media['caption'].get('text') if media.get('caption') is not None else None, 'title': media.get('title'), 'viewer_has_liked': media['has_liked'], 'edge_media_preview_like': {'count': media['like_count']}, 'accessibility_caption': media.get('accessibility_caption'), 'comments': media.get('comment_count'), 'iphone_struct': media}
        with suppress(KeyError):
            fake_node['display_url'] = media['image_versions2']['candidates'][0]['url']
        with suppress(KeyError, TypeError):
            fake_node['video_url'] = media['video_versions'][-1]['url']
            fake_node['video_duration'] = media['video_duration']
            fake_node['video_view_count'] = media['view_count']
        with suppress(KeyError, TypeError):
            fake_node['edge_sidecar_to_children'] = {'edges': [{'node': Post._convert_iphone_carousel(node, media_types)} for node in media['carousel_media']]}
        return cls(context, fake_node, Profile.from_iphone_struct(context, media['user']) if 'user' in media else None)

    @staticmethod
    def _convert_iphone_carousel(iphone_node: Dict[str, Any], media_types: Dict[int, str]) -> Dict[str, Any]:
        fake_node = {'display_url': iphone_node['image_versions2']['candidates'][0]['url'], 'is_video': media_types[iphone_node['media_type']] == 'GraphVideo'}
        if 'video_versions' in iphone_node and iphone_node['video_versions'] is not None:
            fake_node['video_url'] = iphone_node['video_versions'][0]['url']
        return fake_node

    @staticmethod
    def shortcode_to_mediaid(code: str) -> int:
        if len(code) > 11:
            raise InvalidArgumentException('Wrong shortcode "{0}", unable to convert to mediaid.'.format(code))
        code = 'A' * (12 - len(code)) + code
        return int.from_bytes(b64decode(code.encode(), b'-_'), 'big')

    @staticmethod
    def mediaid_to_shortcode(mediaid: int) -> str:
        if mediaid.bit_length() > 64:
            raise InvalidArgumentException('Wrong mediaid {0}, unable to convert to shortcode'.format(str(mediaid)))
        return b64encode(mediaid.to_bytes(9, 'big'), b'-_').decode().replace('A', ' ').lstrip().replace(' ', 'A')

    @staticmethod
    def supported_graphql_types() -> List[str]:
        """The values of __typename fields that the :class:`Post` class can handle."""
        return ['GraphImage', 'GraphVideo', 'GraphSidecar']

    def _asdict(self) -> Dict[str, Any]:
        node = self._node.copy()
        if self._full_metadata_dict:
            node.update(self._full_metadata_dict)
        if self._owner_profile:
            node['owner'] = self.owner_profile._asdict()
        if self._location:
            node['location'] = self._location._asdict()
        if self._iphone_struct_:
            node['iphone_struct'] = self._iphone_struct_
        return node

    @property
    def shortcode(self) -> str:
        """Media shortcode. URL of the post is instagram.com/p/<shortcode>/."""
        return self._node['shortcode'] if 'shortcode' in self._node else self._node['code']

    @property
    def mediaid(self) -> int:
        """The mediaid is a decimal representation of the media shortcode."""
        return int(self._node['id'])

    @property
    def title(self) -> Optional[str]:
        """Title of post"""
        try:
            return self._field('title')
        except KeyError:
            return None

    def __repr__(self) -> str:
        return '<Post {}>'.format(self.shortcode)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Post):
            return self.shortcode == o.shortcode
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.shortcode)

    def _obtain_metadata(self) -> None:
        if not self._full_metadata_dict:
            pic_json = self._context.doc_id_graphql_query('8845758582119845', {'shortcode': self.shortcode})['data']['xdt_shortcode_media']
            if pic_json is None:
                raise BadResponseException('Fetching Post metadata failed.')
            try:
                xdt_types = {'XDTGraphImage': 'GraphImage', 'XDTGraphVideo': 'GraphVideo', 'XDTGraphSidecar': 'GraphSidecar'}
                pic_json['__typename'] = xdt_types[pic_json['__typename']]
            except KeyError as exc:
                raise BadResponseException(f'Unknown __typename in metadata: {pic_json["__typename"]}.') from exc
            self._full_metadata_dict = pic_json
            if self.shortcode != self._full_metadata_dict['shortcode']:
                self._node.update(self._full_metadata_dict)
                raise PostChangedException

    @property
    def _full_metadata(self) -> Dict[str, Any]:
        self._obtain_metadata()
        assert self._full_metadata_dict is not None
        return self._full_metadata_dict

    @property
    def _iphone_struct(self) -> Dict[str, Any]:
        if not self._context.iphone_support:
            raise IPhoneSupportDisabledException('iPhone support is disabled.')
        if not self._context.is_logged_in:
            raise LoginRequiredException('Login required to access iPhone media info endpoint.')
        if not self._iphone_struct_:
            data = self._context.get_iphone_json(path='api/v1/media/{}/info/'.format(self.mediaid), params={})
            self._iphone_struct_ = data['items'][0]
        return self._iphone_struct_

    def _field(self, *keys: str) -> Any:
        """Lookups given fields in _node, and if not found in _full_metadata. Raises KeyError if not found anywhere."""
        try:
            d = self._node
            for key in keys:
                d = d[key]
            return d
        except KeyError:
            d = self._full_metadata
            for key in keys:
                d = d[key]
            return d

    @property
    def owner_profile(self) -> 'Profile':
        """:class:`Profile` instance of the Post's owner."""
        if not self._owner_profile:
            if 'username' in self._node['owner']:
                owner_struct = self._node['owner']
            else:
                owner_struct = self._full_metadata['owner']
            self._owner_profile = Profile(self._context, owner_struct)
        return self._owner_profile

    @property
    def owner_username(self) -> str:
        """The Post's lowercase owner name."""
        return self.owner_profile.username

    @property
    def owner_id(self) -> int:
        """The ID of the Post's owner."""
        if 'owner' in self._node and 'id' in self._node['owner']:
            return self._node['owner']['id']
        else:
            return self.owner_profile.userid

    @property
    def date_local(self) -> datetime:
        """Timestamp when the post was created (local time zone)."""
        return datetime.fromtimestamp(self._get_timestamp_date_created()).astimezone()

    @property
    def date_utc(self) -> datetime:
        """Timestamp when the post was created (UTC)."""
        return datetime.utcfromtimestamp(self._get_timestamp_date_created())

    @property
    def date(self) -> datetime:
        """Synonym to :attr:`~Post.date_utc`"""
        return self.date_utc

    @property
    def profile(self) -> str:
        """Synonym to :attr:`~Post.owner_username`"""
        return self.owner_username

    @property
    def url(self) -> str:
        """URL of the picture / video thumbnail of the post"""
        if self.typename == 'GraphImage' and self._context.iphone_support and self._context.is_logged_in:
            try:
                orig_url = self._iphone_struct['image_versions2']['candidates'][0]['url']
                url = re.sub('([?&])se=\\d+&?', '\\1', orig_url).rstrip('&')
                return url
            except (InstaloaderException, KeyError, IndexError) as err:
                self._context.error(f'Unable to fetch high quality image version of {self}: {err}')
        return self._node['display_url'] if 'display_url' in self._node else self._node['display_src']

    @property
    def typename(self) -> str:
        """Type of post, GraphImage, GraphVideo or GraphSidecar"""
        return self._field('__typename')

    @property
    def mediacount(self) -> int:
        """
        The number of media in a sidecar Post, or 1 if the Post it not a sidecar.
        """
        if self.typename == 'GraphSidecar':
            edges = self._field('edge_sidecar_to_children', 'edges')
            return len(edges)
        return 1

    def _get_timestamp_date_created(self) -> float:
        """Timestamp when the post was created"""
        return self._node['date'] if 'date' in self._node else self._node['taken_at_timestamp']

    def get_is_videos(self) -> List[bool]:
        """
        Return a list containing the ``is_video`` property for each media in the post.
        """
        if self.typename == 'GraphSidecar':
            edges = self._field('edge_sidecar_to_children', 'edges')
            return [edge['node']['is_video'] for edge in edges]
        return [self.is_video]

    def get_sidecar_nodes(self, start: int = 0, end: int = -1) -> Iterator[PostSidecarNode]:
        """
        Sidecar nodes of a Post with typename==GraphSidecar.
        """
        if self.typename == 'GraphSidecar':
            edges = self._field('edge_sidecar_to_children', 'edges')
            if end < 0:
                end = len(edges) - 1
            if start < 0:
                start = len(edges) - 1
            if any((edge['node']['is_video'] and 'video_url' not in edge['node'] for edge in edges[start:end + 1])):
                edges = self._full_metadata['edge_sidecar_to_children']['edges']
            for idx, edge in enumerate(edges):
                if start <= idx <= end:
                    node = edge['node']
                    is_video