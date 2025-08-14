import json
import lzma
import re
from base64 import b64decode, b64encode
from contextlib import suppress
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union
from unicodedata import normalize

from . import __version__
from .exceptions import InvalidArgumentException, BadResponseException, IPhoneSupportDisabledException, LoginRequiredException, PostChangedException, ProfileNotExistsException, QueryReturnedNotFoundException
from .instaloadercontext import InstaloaderContext
from .nodeiterator import FrozenNodeIterator, NodeIterator
from .sectioniterator import SectionIterator

class PostSidecarNode(NamedTuple):
    """Item of a Sidecar Post."""
    is_video: bool
    display_url: str
    video_url: Optional[str]

PostSidecarNode.is_video.__doc__ = "Whether this node is a video."
PostSidecarNode.display_url.__doc__ = "URL of image or video thumbnail."
PostSidecarNode.video_url.__doc__ = "URL of video or None."


class PostCommentAnswer(NamedTuple):
    id: int
    created_at_utc: datetime
    text: str
    owner: "Profile"
    likes_count: int

PostCommentAnswer.id.__doc__ = "ID number of comment."
PostCommentAnswer.created_at_utc.__doc__ = ":class:`~datetime.datetime` when comment was created (UTC)."
PostCommentAnswer.text.__doc__ = "Comment text."
PostCommentAnswer.owner.__doc__ = "Owner :class:`Profile` of the comment."
PostCommentAnswer.likes_count.__doc__ = "Number of likes on comment."


class PostComment:
    def __init__(self, context: "InstaloaderContext", node: Dict[str, Any],
                 answers: Iterator[PostCommentAnswer], post: "Post") -> None:
        self._context: InstaloaderContext = context
        self._node: Dict[str, Any] = node
        self._answers: Iterator[PostCommentAnswer] = answers
        self._post: Post = post

    @classmethod
    def from_iphone_struct(
        cls,
        context: "InstaloaderContext",
        media: Dict[str, Any],
        answers: Iterator[PostCommentAnswer],
        post: "Post",
    ) -> "PostComment":
        return cls(
            context=context,
            node={
                "id": int(media["pk"]),
                "created_at": media["created_at"],
                "text": media["text"],
                "edge_liked_by": {
                    "count": media["comment_like_count"],
                },
                "iphone_struct": media,
            },
            answers=answers,
            post=post,
        )

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
    def owner(self) -> "Profile":
        """ Owner :class:`Profile` of the comment. """
        if "iphone_struct" in self._node:
            return Profile.from_iphone_struct(
                self._context, self._node["iphone_struct"]["user"]
            )
        return Profile(self._context, self._node["owner"])

    @property
    def likes_count(self) -> int:
        """ Number of likes on comment. """
        return self._node.get('edge_liked_by', {}).get('count', 0)

    @property
    def answers(self) -> Iterator[PostCommentAnswer]:
        """ Iterator which yields all :class:`PostCommentAnswer` for the comment. """
        return self._answers

    @property
    def likes(self) -> Iterable["Profile"]:
        """
        Iterate over all likes of a comment. A :class:`Profile` instance of each like is yielded.

        .. versionadded:: 4.11
        """
        if self.likes_count != 0:
            return NodeIterator(
                self._context,
                '5f0b1f6281e72053cbc07909c8d154ae',
                lambda d: d['data']['comment']['edge_liked_by'],
                lambda n: Profile(self._context, n),
                {'comment_id': self.id},
                'https://www.instagram.com/p/{0}/'.format(self._post.shortcode),
            )
        return []

    def __repr__(self) -> str:
        return f'<PostComment {self.id} of {self._post.shortcode}>'


class PostLocation(NamedTuple):
    id: int
    name: str
    slug: str
    has_public_page: Optional[bool]
    lat: Optional[float]
    lng: Optional[float]

PostLocation.id.__doc__ = "ID number of location."
PostLocation.name.__doc__ = "Location name."
PostLocation.slug.__doc__ = "URL friendly variant of location name."
PostLocation.has_public_page.__doc__ = "Whether location has a public page."
PostLocation.lat.__doc__ = "Latitude (:class:`float` or None)."
PostLocation.lng.__doc__ = "Longitude (:class:`float` or None)."

_hashtag_regex = re.compile(r"(?:#)((?:\w){1,150})")
_mention_regex = re.compile(r"(?:^|[^\w\n]|_)(?:@)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)", re.ASCII)

def _optional_normalize(string: Optional[str]) -> Optional[str]:
    if string is not None:
        return normalize("NFC", string)
    else:
        return None

class Post:
    """
    Structure containing information about an Instagram post.
    """
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any],
                 owner_profile: Optional["Profile"] = None) -> None:
        assert 'shortcode' in node or 'code' in node
        self._context: InstaloaderContext = context
        self._node: Dict[str, Any] = node
        self._owner_profile: Optional[Profile] = owner_profile
        self._full_metadata_dict: Optional[Dict[str, Any]] = None
        self._location: Optional[PostLocation] = None
        self._iphone_struct_ : Optional[Any] = None
        if 'iphone_struct' in node:
            self._iphone_struct_ = node['iphone_struct']

    @classmethod
    def from_shortcode(cls, context: InstaloaderContext, shortcode: str) -> "Post":
        post = cls(context, {'shortcode': shortcode})
        post._node = post._full_metadata
        return post

    @classmethod
    def from_mediaid(cls, context: InstaloaderContext, mediaid: int) -> "Post":
        return cls.from_shortcode(context, Post.mediaid_to_shortcode(mediaid))

    @classmethod
    def from_iphone_struct(cls, context: InstaloaderContext, media: Dict[str, Any]) -> "Post":
        media_types: Dict[int, str] = {
            1: "GraphImage",
            2: "GraphVideo",
            8: "GraphSidecar",
        }
        fake_node: Dict[str, Any] = {
            "shortcode": media["code"],
            "id": media["pk"],
            "__typename": media_types[media["media_type"]],
            "is_video": media_types[media["media_type"]] == "GraphVideo",
            "date": media["taken_at"],
            "caption": media["caption"].get("text") if media.get("caption") is not None else None,
            "title": media.get("title"),
            "viewer_has_liked": media["has_liked"],
            "edge_media_preview_like": {"count": media["like_count"]},
            "accessibility_caption": media.get("accessibility_caption"),
            "comments": media.get("comment_count"),
            "iphone_struct": media,
        }
        with suppress(KeyError):
            fake_node["display_url"] = media['image_versions2']['candidates'][0]['url']
        with suppress(KeyError, TypeError):
            fake_node["video_url"] = media['video_versions'][-1]['url']
            fake_node["video_duration"] = media["video_duration"]
            fake_node["video_view_count"] = media["view_count"]
        with suppress(KeyError, TypeError):
            fake_node["edge_sidecar_to_children"] = {"edges": [{"node":
                Post._convert_iphone_carousel(node, media_types)}
                for node in media["carousel_media"]]}
        owner: Optional["Profile"] = Profile.from_iphone_struct(context, media["user"]) if "user" in media else None
        return cls(context, fake_node, owner)

    @staticmethod
    def _convert_iphone_carousel(iphone_node: Dict[str, Any], media_types: Dict[int, str]) -> Dict[str, Any]:
        fake_node: Dict[str, Any] = {
            "display_url": iphone_node["image_versions2"]["candidates"][0]["url"],
            "is_video": media_types[iphone_node["media_type"]] == "GraphVideo",
        }
        if "video_versions" in iphone_node and iphone_node["video_versions"] is not None:
            fake_node["video_url"] = iphone_node["video_versions"][0]["url"]
        return fake_node

    @staticmethod
    def shortcode_to_mediaid(code: str) -> int:
        if len(code) > 11:
            raise InvalidArgumentException(f"Wrong shortcode \"{code}\", unable to convert to mediaid.")
        code = 'A' * (12 - len(code)) + code
        return int.from_bytes(b64decode(code.encode(), b'-_'), 'big')

    @staticmethod
    def mediaid_to_shortcode(mediaid: int) -> str:
        if mediaid.bit_length() > 64:
            raise InvalidArgumentException(f"Wrong mediaid {mediaid}, unable to convert to shortcode")
        encoded: str = b64encode(mediaid.to_bytes(9, 'big'), b'-_').decode()
        return encoded.replace('A', ' ').lstrip().replace(' ', 'A')

    @staticmethod
    def supported_graphql_types() -> List[str]:
        return ["GraphImage", "GraphVideo", "GraphSidecar"]

    def _asdict(self) -> Dict[str, Any]:
        node: Dict[str, Any] = self._node.copy()
        if self._full_metadata_dict:
            node.update(self._full_metadata_dict)
        if self._owner_profile:
            node['owner'] = self.owner_profile._asdict()
        if self._location:
            node['location'] = self._location._asdict()  # type: ignore[attr-defined]
        if self._iphone_struct_:
            node['iphone_struct'] = self._iphone_struct_
        return node

    @property
    def shortcode(self) -> str:
        return self._node['shortcode'] if 'shortcode' in self._node else self._node['code']

    @property
    def mediaid(self) -> int:
        return int(self._node['id'])

    @property
    def title(self) -> Optional[str]:
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
            pic_json: Dict[str, Any] = self._context.doc_id_graphql_query(
                "8845758582119845", {"shortcode": self.shortcode}
            )["data"]["xdt_shortcode_media"]
            if pic_json is None:
                raise BadResponseException("Fetching Post metadata failed.")
            try:
                xdt_types: Dict[str, str] = {
                    "XDTGraphImage": "GraphImage",
                    "XDTGraphVideo": "GraphVideo",
                    "XDTGraphSidecar": "GraphSidecar",
                }
                pic_json["__typename"] = xdt_types[pic_json["__typename"]]
            except KeyError as exc:
                raise BadResponseException(
                    f"Unknown __typename in metadata: {pic_json['__typename']}."
                ) from exc
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
            raise IPhoneSupportDisabledException("iPhone support is disabled.")
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to access iPhone media info endpoint.")
        if not self._iphone_struct_:
            data: Dict[str, Any] = self._context.get_iphone_json(path=f'api/v1/media/{self.mediaid}/info/', params={})
            self._iphone_struct_ = data['items'][0]
        return self._iphone_struct_

    def _field(self, *keys: str) -> Any:
        try:
            d: Any = self._node
            for key in keys:
                d = d[key]
            return d
        except KeyError:
            d = self._full_metadata
            for key in keys:
                d = d[key]
            return d

    @property
    def owner_profile(self) -> "Profile":
        if not self._owner_profile:
            if 'username' in self._node['owner']:
                owner_struct = self._node['owner']
            else:
                owner_struct = self._full_metadata['owner']
            self._owner_profile = Profile(self._context, owner_struct)
        return self._owner_profile

    @property
    def owner_username(self) -> str:
        return self.owner_profile.username

    @property
    def owner_id(self) -> int:
        if 'owner' in self._node and 'id' in self._node['owner']:
            return self._node['owner']['id']
        else:
            return self.owner_profile.userid

    @property
    def date_local(self) -> datetime:
        return datetime.fromtimestamp(self._get_timestamp_date_created()).astimezone()

    @property
    def date_utc(self) -> datetime:
        return datetime.utcfromtimestamp(self._get_timestamp_date_created())

    @property
    def date(self) -> datetime:
        return self.date_utc

    @property
    def profile(self) -> str:
        return self.owner_username

    @property
    def url(self) -> str:
        if self.typename == "GraphImage" and self._context.iphone_support and self._context.is_logged_in:
            try:
                orig_url: str = self._iphone_struct['image_versions2']['candidates'][0]['url']
                url: str = re.sub(r'([?&])se=\d+&?', r'\1', orig_url).rstrip('&')
                return url
            except (BadResponseException, KeyError, IndexError) as err:
                self._context.error(f"Unable to fetch high quality image version of {self}: {err}")
        return self._node["display_url"] if "display_url" in self._node else self._node["display_src"]

    @property
    def typename(self) -> str:
        return self._field('__typename')

    @property
    def mediacount(self) -> int:
        if self.typename == 'GraphSidecar':
            edges = self._field('edge_sidecar_to_children', 'edges')
            return len(edges)
        return 1

    def _get_timestamp_date_created(self) -> float:
        return (self._node["date"]
                if "date" in self._node
                else self._node["taken_at_timestamp"])

    def get_is_videos(self) -> List[bool]:
        if self.typename == 'GraphSidecar':
            edges = self._field('edge_sidecar_to_children', 'edges')
            return [edge['node']['is_video'] for edge in edges]
        return [self.is_video]

    def get_sidecar_nodes(self, start: int = 0, end: int = -1) -> Iterator[PostSidecarNode]:
        if self.typename == 'GraphSidecar':
            edges = self._field('edge_sidecar_to_children', 'edges')
            if end < 0:
                end = len(edges)-1
            if start < 0:
                start = len(edges)-1
            if any(edge['node']['is_video'] and 'video_url' not in edge['node'] for edge in edges[start:(end+1)]):
                edges = self._full_metadata['edge_sidecar_to_children']['edges']
            for idx, edge in enumerate(edges):
                if start <= idx <= end:
                    node = edge['node']
                    is_video: bool = node['is_video']
                    display_url: str = node['display_url']
                    if not is_video and self._context.iphone_support and self._context.is_logged_in:
                        try:
                            carousel_media = self._iphone_struct['carousel_media']
                            orig_url = carousel_media[idx]['image_versions2']['candidates'][0]['url']
                            display_url = re.sub(r'([?&])se=\d+&?', r'\1', orig_url).rstrip('&')
                        except (BadResponseException, KeyError, IndexError) as err:
                            self._context.error(f"Unable to fetch high quality image version of {self}: {err}")
                    yield PostSidecarNode(is_video=is_video, display_url=display_url,
                                          video_url=node['video_url'] if is_video else None)
    @property
    def caption(self) -> Optional[str]:
        if "edge_media_to_caption" in self._node and self._node["edge_media_to_caption"]["edges"]:
            return _optional_normalize(self._node["edge_media_to_caption"]["edges"][0]["node"]["text"])
        elif "caption" in self._node:
            return _optional_normalize(self._node["caption"])
        return None

    @property
    def caption_hashtags(self) -> List[str]:
        if not self.caption:
            return []
        return _hashtag_regex.findall(self.caption.lower())

    @property
    def caption_mentions(self) -> List[str]:
        if not self.caption:
            return []
        return _mention_regex.findall(self.caption.lower())

    @property
    def pcaption(self) -> str:
        def _elliptify(caption: str) -> str:
            pcaption: str = ' '.join([s.replace('/', '\u2215') for s in caption.splitlines() if s]).strip()
            return (pcaption[:30] + "\u2026") if len(pcaption) > 31 else pcaption
        return _elliptify(self.caption) if self.caption else ''

    @property
    def accessibility_caption(self) -> Optional[str]:
        try:
            return self._field("accessibility_caption")
        except KeyError:
            return None

    @property
    def tagged_users(self) -> List[str]:
        try:
            return [edge['node']['user']['username'].lower() for edge in self._field('edge_media_to_tagged_user', 'edges')]
        except KeyError:
            return []

    @property
    def is_video(self) -> bool:
        return self._node['is_video']

    @property
    def video_url(self) -> Optional[str]:
        if self.is_video:
            version_urls: List[str] = []
            try:
                version_urls.append(self._field('video_url'))
            except (BadResponseException, KeyError, IndexError) as err:
                self._context.error(f"Warning: Unable to fetch video from graphql of {self}: {err}")
            if self._context.iphone_support and self._context.is_logged_in:
                try:
                    version_urls.extend(version['url'] for version in self._iphone_struct['video_versions'])
                except (BadResponseException, KeyError, IndexError) as err:
                    self._context.error(f"Unable to fetch high-quality video version of {self}: {err}")
            version_urls = list(dict.fromkeys(version_urls))
            if len(version_urls) == 0:
                return None
            if len(version_urls) == 1:
                return version_urls[0]
            url_candidates: List[Tuple[int, str]] = []
            for idx, version_url in enumerate(version_urls):
                try:
                    content_length: int = int(self._context.head(version_url, allow_redirects=True).headers.get('Content-Length', 0))
                    url_candidates.append((content_length, version_url))
                except (BadResponseException, KeyError, IndexError) as err:
                    self._context.error(f"Video URL candidate {idx+1}/{len(version_urls)} for {self}: {err}")
            if not url_candidates:
                return version_urls[0]
            url_candidates.sort()
            return url_candidates[-1][1]
        return None

    @property
    def video_view_count(self) -> Optional[int]:
        if self.is_video:
            return self._field('video_view_count')
        return None

    @property
    def video_duration(self) -> Optional[float]:
        if self.is_video:
            return self._field('video_duration')
        return None

    @property
    def viewer_has_liked(self) -> Optional[bool]:
        if not self._context.is_logged_in:
            return None
        if 'likes' in self._node and 'viewer_has_liked' in self._node['likes']:
            return self._node['likes']['viewer_has_liked']
        return self._field('viewer_has_liked')

    @property
    def likes(self) -> int:
        return self._field('edge_media_preview_like', 'count')

    @property
    def comments(self) -> int:
        comments_data = self._node.get('edge_media_to_comment')
        if comments_data and 'count' in comments_data:
            return comments_data['count']
        try:
            return self._field('edge_media_to_parent_comment', 'count')
        except KeyError:
            return self._field('edge_media_to_comment', 'count')

    def _get_comments_via_iphone_endpoint(self) -> Iterable[PostComment]:
        def _query(min_id: Optional[str] = None) -> Dict[str, Any]:
            pagination_params: Dict[str, str] = {"min_id": min_id} if min_id is not None else {}
            return self._context.get_iphone_json(
                f"api/v1/media/{self.mediaid}/comments/",
                {"can_support_threading": "true", "permalink_enabled": "false", **pagination_params},
            )

        def _answers(comment_node: Dict[str, Any]) -> Iterator[PostCommentAnswer]:
            def _answer(child_comment: Dict[str, Any]) -> PostCommentAnswer:
                return PostCommentAnswer(
                    id=int(child_comment["pk"]),
                    created_at_utc=datetime.utcfromtimestamp(child_comment["created_at"]),
                    text=child_comment["text"],
                    owner=Profile.from_iphone_struct(self._context, child_comment["user"]),
                    likes_count=child_comment["comment_like_count"],
                )
            child_comment_count: int = comment_node["child_comment_count"]
            if child_comment_count == 0:
                return iter([])
            preview_child_comments: List[Dict[str, Any]] = comment_node["preview_child_comments"]
            if child_comment_count == len(preview_child_comments):
                return (_answer(child_comment) for child_comment in preview_child_comments)
            pk: str = comment_node["pk"]
            answers_json: Dict[str, Any] = self._context.get_iphone_json(
                f"api/v1/media/{self.mediaid}/comments/{pk}/child_comments/", {"max_id": ""}
            )
            return (_answer(child_comment) for child_comment in answers_json["child_comments"])

        def _paginated_comments(comments_json: Dict[str, Any]) -> Iterator[PostComment]:
            for comment_node in comments_json.get("comments", []):
                yield PostComment.from_iphone_struct(
                    self._context, comment_node, _answers(comment_node), self
                )
            next_min_id: Optional[str] = comments_json.get("next_min_id")
            if next_min_id:
                yield from _paginated_comments(_query(next_min_id))

        return _paginated_comments(_query())

    def get_comments(self) -> Iterable[PostComment]:
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to access comments of a post.")

        def _postcommentanswer(node: Dict[str, Any]) -> PostCommentAnswer:
            return PostCommentAnswer(
                id=int(node['id']),
                created_at_utc=datetime.utcfromtimestamp(node['created_at']),
                text=node['text'],
                owner=Profile(self._context, node['owner']),
                likes_count=node.get('edge_liked_by', {}).get('count', 0)
            )

        def _postcommentanswers(node: Dict[str, Any]) -> Iterator[PostCommentAnswer]:
            if 'edge_threaded_comments' not in node:
                return iter([])
            answer_count: int = node['edge_threaded_comments']['count']
            if answer_count == 0:
                return iter([])
            answer_edges = node['edge_threaded_comments']['edges']
            if answer_count == len(answer_edges):
                return (_postcommentanswer(comment['node']) for comment in answer_edges)
            return NodeIterator(
                self._context,
                '51fdd02b67508306ad4484ff574a0b62',
                lambda d: d['data']['comment']['edge_threaded_comments'],
                _postcommentanswer,
                {'comment_id': node['id']},
                f'https://www.instagram.com/p/{self.shortcode}/'
            )

        def _postcomment(node: Dict[str, Any]) -> PostComment:
            return PostComment(context=self._context, node=node, answers=_postcommentanswers(node), post=self)
        if self.comments == 0:
            return []
        try:
            comment_edges = self._field("edge_media_to_parent_comment", "edges")
        except KeyError:
            comment_edges = self._field("edge_media_to_comment", "edges")
        answers_count: int = sum(edge['node'].get('edge_threaded_comments', {}).get('count', 0) for edge in comment_edges)
        if self.comments == len(comment_edges) + answers_count:
            return [_postcomment(comment['node']) for comment in comment_edges]
        if self.comments > NodeIterator.page_length():
            return self._get_comments_via_iphone_endpoint()
        return NodeIterator(
            self._context,
            '97b41c52301f77ce508f55e66d17620e',
            lambda d: d['data']['shortcode_media']['edge_media_to_parent_comment'],
            _postcomment,
            {'shortcode': self.shortcode},
            f'https://www.instagram.com/p/{self.shortcode}/'
        )

    def get_likes(self) -> Iterator["Profile"]:
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to access likes of a post.")
        if self.likes == 0:
            return iter([])
        likes_edges = self._field('edge_media_preview_like', 'edges')
        if self.likes == len(likes_edges):
            return (Profile(self._context, like['node']) for like in likes_edges)
        return NodeIterator(
            self._context,
            '1cb6ec562846122743b61e492c85999f',
            lambda d: d['data']['shortcode_media']['edge_liked_by'],
            lambda n: Profile(self._context, n),
            {'shortcode': self.shortcode},
            f'https://www.instagram.com/p/{self.shortcode}/'
        )

    @property
    def is_sponsored(self) -> bool:
        try:
            sponsor_edges = self._field('edge_media_to_sponsor_user', 'edges')
        except KeyError:
            return False
        return bool(sponsor_edges)

    @property
    def sponsor_users(self) -> List["Profile"]:
        return ([] if not self.is_sponsored else
                [Profile(self._context, edge['node']['sponsor']) for edge in
                 self._field('edge_media_to_sponsor_user', 'edges')])

    @property
    def location(self) -> Optional[PostLocation]:
        loc: Optional[Dict[str, Any]] = self._field("location")
        if self._location or not loc:
            return self._location
        if not self._context.is_logged_in:
            return None
        location_id: int = int(loc['id'])
        if any(k not in loc for k in ('name', 'slug', 'has_public_page', 'lat', 'lng')):
            loc.update(self._context.get_json(f"explore/locations/{location_id}/", params={'__a': 1, '__d': 'dis'})['native_location_data']['location_info'])
        self._location = PostLocation(location_id, loc['name'], loc['slug'], loc['has_public_page'], loc.get('lat'), loc.get('lng'))
        return self._location

    @property
    def is_pinned(self) -> bool:
        return 'pinned_for_users' in self._node and bool(self._node['pinned_for_users'])


class Profile:
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any]) -> None:
        assert 'username' in node
        self._context: InstaloaderContext = context
        self._has_public_story: Optional[bool] = None
        self._node: Dict[str, Any] = node
        self._has_full_metadata: bool = False
        self._iphone_struct_: Optional[Any] = None
        if 'iphone_struct' in node:
            self._iphone_struct_ = node['iphone_struct']

    @classmethod
    def from_username(cls, context: InstaloaderContext, username: str) -> "Profile":
        profile = cls(context, {'username': username.lower()})
        profile._obtain_metadata()
        return profile

    @classmethod
    def from_id(cls, context: InstaloaderContext, profile_id: int) -> "Profile":
        if profile_id in context.profile_id_cache:
            return context.profile_id_cache[profile_id]
        data: Dict[str, Any] = context.graphql_query('7c16654f22c819fb63d1183034a5162f',
                                                      {'user_id': str(profile_id),
                                                       'include_chaining': False,
                                                       'include_reel': True,
                                                       'include_suggested_users': False,
                                                       'include_logged_out_extras': False,
                                                       'include_highlight_reels': False})['data']['user']
        if data:
            profile = cls(context, data['reel']['owner'])
        else:
            raise ProfileNotExistsException("No profile found, the user may have blocked you (ID: " + str(profile_id) + ").")
        context.profile_id_cache[profile_id] = profile
        return profile

    @classmethod
    def from_iphone_struct(cls, context: InstaloaderContext, media: Dict[str, Any]) -> "Profile":
        return cls(context, {
            "id": media["pk"],
            "username": media["username"],
            "is_private": media["is_private"],
            "full_name": media["full_name"],
            "profile_pic_url_hd": media["profile_pic_url"],
            "iphone_struct": media,
        })

    @classmethod
    def own_profile(cls, context: InstaloaderContext) -> "Profile":
        if not context.is_logged_in:
            raise LoginRequiredException("Login required to access own profile.")
        return cls(context, context.graphql_query("d6f4427fbe92d846298cf93df0b937d3", {})["data"]["user"])

    def _asdict(self) -> Dict[str, Any]:
        json_node: Dict[str, Any] = self._node.copy()
        json_node.pop('edge_media_collections', None)
        json_node.pop('edge_owner_to_timeline_media', None)
        json_node.pop('edge_saved_media', None)
        json_node.pop('edge_felix_video_timeline', None)
        if self._iphone_struct_:
            json_node['iphone_struct'] = self._iphone_struct_
        return json_node

    def _obtain_metadata(self) -> None:
        try:
            if not self._has_full_metadata:
                metadata: Dict[str, Any] = self._context.get_iphone_json(f'api/v1/users/web_profile_info/?username={self.username}', params={})
                if metadata['data']['user'] is None:
                    raise ProfileNotExistsException(f'Profile {self.username} does not exist.')
                self._node = metadata['data']['user']
                self._has_full_metadata = True
        except (QueryReturnedNotFoundException, KeyError) as err:
            top_search_results = TopSearchResults(self._context, self.username)
            similar_profiles: List[str] = [profile.username for profile in top_search_results.get_profiles()]
            if similar_profiles:
                if self.username in similar_profiles:
                    raise ProfileNotExistsException(f"Profile {self.username} seems to exist, but could not be loaded.") from err
                raise ProfileNotExistsException(f'Profile {self.username} does not exist.\nThe most similar profile{"s" if len(similar_profiles)>1 else ""}: {", ".join(similar_profiles[0:5])}.') from err
            raise ProfileNotExistsException(f'Profile {self.username} does not exist.') from err

    def _metadata(self, *keys: str) -> Any:
        try:
            d: Any = self._node
            for key in keys:
                d = d[key]
            return d
        except KeyError:
            self._obtain_metadata()
            d = self._node
            for key in keys:
                d = d[key]
            return d

    @property
    def _iphone_struct(self) -> Dict[str, Any]:
        if not self._context.iphone_support:
            raise IPhoneSupportDisabledException("iPhone support is disabled.")
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to access iPhone profile info endpoint.")
        if not self._iphone_struct_:
            data: Dict[str, Any] = self._context.get_iphone_json(path=f'api/v1/users/{self.userid}/info/', params={})
            self._iphone_struct_ = data['user']
        return self._iphone_struct_

    @property
    def userid(self) -> int:
        return int(self._metadata('id'))

    @property
    def username(self) -> str:
        return self._metadata('username').lower()

    def __repr__(self) -> str:
        return '<Profile {} ({})>'.format(self.username, self.userid)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Profile):
            return self.userid == o.userid
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.userid)

    @property
    def is_private(self) -> bool:
        return self._metadata('is_private')

    @property
    def followed_by_viewer(self) -> bool:
        return self._metadata('followed_by_viewer')

    @property
    def mediacount(self) -> int:
        return self._metadata('edge_owner_to_timeline_media', 'count')

    @property
    def igtvcount(self) -> int:
        return self._metadata('edge_felix_video_timeline', 'count')

    @property
    def followers(self) -> int:
        return self._metadata('edge_followed_by', 'count')

    @property
    def followees(self) -> int:
        return self._metadata('edge_follow', 'count')

    @property
    def external_url(self) -> Optional[str]:
        return self._metadata('external_url')

    @property
    def is_business_account(self) -> bool:
        return self._metadata('is_business_account')

    @property
    def business_category_name(self) -> str:
        return self._metadata('business_category_name')

    @property
    def biography(self) -> str:
        return normalize("NFC", self._metadata('biography'))

    @property
    def biography_hashtags(self) -> List[str]:
        if not self.biography:
            return []
        return _hashtag_regex.findall(self.biography.lower())

    @property
    def biography_mentions(self) -> List[str]:
        if not self.biography:
            return []
        return _mention_regex.findall(self.biography.lower())

    @property
    def blocked_by_viewer(self) -> bool:
        return self._metadata('blocked_by_viewer')

    @property
    def follows_viewer(self) -> bool:
        return self._metadata('follows_viewer')

    @property
    def full_name(self) -> str:
        return self._metadata('full_name')

    @property
    def has_blocked_viewer(self) -> bool:
        return self._metadata('has_blocked_viewer')

    @property
    def has_highlight_reels(self) -> bool:
        return True

    @property
    def has_public_story(self) -> bool:
        if self._has_public_story is None:
            self._obtain_metadata()
            data: Dict[str, Any] = self._context.graphql_query('9ca88e465c3f866a76f7adee3871bdd8',
                                                                {'user_id': self.userid, 'include_chaining': False,
                                                                 'include_reel': False, 'include_suggested_users': False,
                                                                 'include_logged_out_extras': True,
                                                                 'include_highlight_reels': False},
                                                                f'https://www.instagram.com/{self.username}/')
            self._has_public_story = data['data']['user']['has_public_story']
        assert self._has_public_story is not None
        return self._has_public_story

    @property
    def has_viewable_story(self) -> bool:
        return self.has_public_story or self.followed_by_viewer and self.has_highlight_reels

    @property
    def has_requested_viewer(self) -> bool:
        return self._metadata('has_requested_viewer')

    @property
    def is_verified(self) -> bool:
        return self._metadata('is_verified')

    @property
    def requested_by_viewer(self) -> bool:
        return self._metadata('requested_by_viewer')

    @property
    def profile_pic_url(self) -> str:
        if self._context.iphone_support and self._context.is_logged_in:
            try:
                return self._iphone_struct['hd_profile_pic_url_info']['url']
            except (BadResponseException, KeyError) as err:
                self._context.error(f"Unable to fetch high quality profile pic: {err}")
                return self._metadata("profile_pic_url_hd")
        else:
            return self._metadata("profile_pic_url_hd")

    @property
    def profile_pic_url_no_iphone(self) -> str:
        return self._metadata("profile_pic_url_hd")

    def get_profile_pic_url(self) -> str:
        return self.profile_pic_url

    def get_posts(self) -> NodeIterator[Post]:
        self._obtain_metadata()
        return NodeIterator(
            context = self._context,
            edge_extractor = lambda d: d['data']['xdt_api__v1__feed__user_timeline_graphql_connection'],
            node_wrapper = lambda n: Post.from_iphone_struct(self._context, n),
            query_variables = {'data': {
                'count': 12, 'include_relationship_info': True,
                'latest_besties_reel_media': True, 'latest_reel_media': True},
             'username': self.username},
            query_referer = f'https://www.instagram.com/{self.username}/',
            is_first = Profile._make_is_newest_checker(),
            doc_id = '7898261790222653',
            query_hash = None,
        )

    def get_saved_posts(self) -> NodeIterator[Post]:
        if self.username != self._context.username:
            raise LoginRequiredException(f"Login as {self.username} required to get that profile's saved posts.")
        return NodeIterator(
            self._context,
            'f883d95537fbcd400f466f63d42bd8a1',
            lambda d: d['data']['user']['edge_saved_media'],
            lambda n: Post(self._context, n),
            {'id': self.userid},
            f'https://www.instagram.com/{self.username}/'
        )

    def get_tagged_posts(self) -> NodeIterator[Post]:
        self._obtain_metadata()
        return NodeIterator(
            self._context,
            'e31a871f7301132ceaab56507a66bbb7',
            lambda d: d['data']['user']['edge_user_to_photos_of_you'],
            lambda n: Post(self._context, n, self if int(n['owner']['id']) == self.userid else None),
            {'id': self.userid},
            f'https://www.instagram.com/{self.username}/',
            is_first=Profile._make_is_newest_checker()
        )

    def get_reels(self) -> NodeIterator[Post]:
        self._obtain_metadata()
        return NodeIterator(
            context = self._context,
            edge_extractor = lambda d: d['data']['xdt_api__v1__clips__user__connection_v2'],
            node_wrapper = lambda n: Post.from_shortcode(context=self._context, shortcode=n["media"]["code"]),
            query_variables = {'data': {
                'page_size': 12, 'include_feed_video': True, "target_user_id": str(self.userid)}},
            query_referer = f'https://www.instagram.com/{self.username}/',
            is_first = Profile._make_is_newest_checker(),
            doc_id = '7845543455542541',
            query_hash = None,
        )

    def get_igtv_posts(self) -> NodeIterator[Post]:
        self._obtain_metadata()
        return NodeIterator(
            self._context,
            'bc78b344a68ed16dd5d7f264681c4c76',
            lambda d: d['data']['user']['edge_felix_video_timeline'],
            lambda n: Post(self._context, n, self),
            {'id': self.userid},
            f'https://www.instagram.com/{self.username}/channel/',
            self._metadata('edge_felix_video_timeline'),
            Profile._make_is_newest_checker()
        )

    @staticmethod
    def _make_is_newest_checker() -> Callable[[Post, Optional[Post]], bool]:
        return lambda post, first: first is None or post.date_local > first.date_local

    def get_followed_hashtags(self) -> NodeIterator["Hashtag"]:
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to get a profile's followers.")
        self._obtain_metadata()
        return NodeIterator(
            self._context,
            'e6306cc3dbe69d6a82ef8b5f8654c50b',
            lambda d: d["data"]["user"]["edge_following_hashtag"],
            lambda n: Hashtag(self._context, n),
            {'id': str(self.userid)},
            f'https://www.instagram.com/{self.username}/'
        )

    def get_followers(self) -> NodeIterator["Profile"]:
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to get a profile's followers.")
        self._obtain_metadata()
        return NodeIterator(
            self._context,
            '37479f2b8209594dde7facb0d904896a',
            lambda d: d['data']['user']['edge_followed_by'],
            lambda n: Profile(self._context, n),
            {'id': str(self.userid)},
            f'https://www.instagram.com/{self.username}/'
        )

    def get_followees(self) -> NodeIterator["Profile"]:
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to get a profile's followees.")
        self._obtain_metadata()
        return NodeIterator(
            self._context,
            '58712303d941c6855d4e888c5f0cd22f',
            lambda d: d['data']['user']['edge_follow'],
            lambda n: Profile(self._context, n),
            {'id': str(self.userid)},
            f'https://www.instagram.com/{self.username}/'
        )

    def get_similar_accounts(self) -> Iterator["Profile"]:
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to get a profile's similar accounts.")
        self._obtain_metadata()
        yield from (Profile(self._context, edge["node"]) for edge in
                    self._context.graphql_query("ad99dd9d3646cc3c0dda65debcd266a7",
                                                {"user_id": str(self.userid), "include_chaining": True},
                                                f'https://www.instagram.com/{self.username}/'
                                                )["data"]["user"]["edge_chaining"]["edges"])

class StoryItem:
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any], owner_profile: Optional[Profile] = None) -> None:
        self._context: InstaloaderContext = context
        self._node: Dict[str, Any] = node
        self._owner_profile: Optional[Profile] = owner_profile
        self._iphone_struct_: Optional[Any] = None
        if 'iphone_struct' in node:
            self._iphone_struct_ = node['iphone_struct']

    def _asdict(self) -> Dict[str, Any]:
        node = self._node.copy()
        if self._owner_profile:
            node['owner'] = self._owner_profile._asdict()
        if self._iphone_struct_:
            node['iphone_struct'] = self._iphone_struct_
        return node

    @property
    def mediaid(self) -> int:
        return int(self._node['id'])

    @property
    def shortcode(self) -> str:
        return Post.mediaid_to_shortcode(self.mediaid)

    def __repr__(self) -> str:
        return '<StoryItem {}>'.format(self.mediaid)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, StoryItem):
            return self.mediaid == o.mediaid
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.mediaid)

    @classmethod
    def from_mediaid(cls, context: InstaloaderContext, mediaid: int) -> "StoryItem":
        pic_json: Dict[str, Any] = context.graphql_query(
            '2b0673e0dc4580674a88d426fe00ea90',
            {'shortcode': Post.mediaid_to_shortcode(mediaid)}
        )
        shortcode_media: Optional[Dict[str, Any]] = pic_json['data']['shortcode_media']
        if shortcode_media is None:
            raise BadResponseException("Fetching StoryItem metadata failed.")
        return cls(context, shortcode_media)

    @property
    def _iphone_struct(self) -> Dict[str, Any]:
        if not self._context.iphone_support:
            raise IPhoneSupportDisabledException("iPhone support is disabled.")
        if not self._context.is_logged_in:
            raise LoginRequiredException("Login required to access iPhone media info endpoint.")
        if not self._iphone_struct_:
            data: Dict[str, Any] = self._context.get_iphone_json(
                path=f'api/v1/feed/reels_media/?reel_ids={self.owner_id}', params={}
            )
            self._iphone_struct_ = {}
            for item in data['reels'][str(self.owner_id)]['items']:
                if item['pk'] == self.mediaid:
                    self._iphone_struct_ = item
                    break
        return self._iphone_struct_

    @property
    def owner_profile(self) -> Profile:
        if not self._owner_profile:
            self._owner_profile = Profile.from_id(self._context, self._node['owner']['id'])
        return self._owner_profile

    @property
    def owner_username(self) -> str:
        return self.owner_profile.username

    @property
    def owner_id(self) -> int:
        return self.owner_profile.userid

    @property
    def date_local(self) -> datetime:
        return datetime.fromtimestamp(self._node['taken_at_timestamp']).astimezone()

    @property
    def date_utc(self) -> datetime:
        return datetime.utcfromtimestamp(self._node['taken_at_timestamp'])

    @property
    def date(self) -> datetime:
        return self.date_utc

    @property
    def profile(self) -> str:
        return self.owner_username

    @property
    def expiring_local(self) -> datetime:
        return datetime.fromtimestamp(self._node['expiring_at_timestamp'])

    @property
    def expiring_utc(self) -> datetime:
        return datetime.utcfromtimestamp(self._node['expiring_at_timestamp'])

    @property
    def url(self) -> str:
        if self.typename in ["GraphStoryImage", "StoryImage"] and self._context.iphone_support and self._context.is_logged_in:
            try:
                orig_url: str = self._iphone_struct['image_versions2']['candidates'][0]['url']
                url: str = re.sub(r'([?&])se=\d+&?', r'\1', orig_url).rstrip('&')
                return url
            except (BadResponseException, KeyError, IndexError) as err:
                self._context.error(f"Unable to fetch high quality image version of {self}: {err}")
        return self._node['display_resources'][-1]['src']

    @property
    def typename(self) -> str:
        return self._node['__typename']

    @property
    def caption(self) -> Optional[str]:
        if "edge_media_to_caption" in self._node and self._node["edge_media_to_caption"]["edges"]:
            return _optional_normalize(self._node["edge_media_to_caption"]["edges"][0]["node"]["text"])
        elif "caption" in self._node:
            return _optional_normalize(self._node["caption"])
        return None

    @property
    def caption_hashtags(self) -> List[str]:
        if not self.caption:
            return []
        return _hashtag_regex.findall(self.caption.lower())

    @property
    def caption_mentions(self) -> List[str]:
        if not self.caption:
            return []
        return _mention_regex.findall(self.caption.lower())

    @property
    def pcaption(self) -> str:
        def _elliptify(caption: str) -> str:
            pcaption: str = ' '.join([s.replace('/', '\u2215') for s in caption.splitlines() if s]).strip()
            return (pcaption[:30] + "\u2026") if len(pcaption) > 31 else pcaption
        return _elliptify(self.caption) if self.caption else ''

    @property
    def is_video(self) -> bool:
        return self._node['is_video']

    @property
    def video_url(self) -> Optional[str]:
        if self.is_video:
            version_urls: List[str] = []
            try:
                version_urls.append(self._node['video_resources'][-1]['src'])
            except (BadResponseException, KeyError, IndexError) as err:
                self._context.error(f"Warning: Unable to fetch video from graphql of {self}: {err}")
            if self._context.iphone_support and self._context.is_logged_in:
                try:
                    version_urls.extend(version['url'] for version in self._iphone_struct['video_versions'])
                except (BadResponseException, KeyError, IndexError) as err:
                    self._context.error(f"Unable to fetch high-quality video version of {self}: {err}")
            version_urls = list(dict.fromkeys(version_urls))
            if len(version_urls) == 0:
                return None
            if len(version_urls) == 1:
                return version_urls[0]
            url_candidates: List[Tuple[int, str]] = []
            for idx, version_url in enumerate(version_urls):
                try:
                    content_length = int(self._context.head(version_url, allow_redirects=True).headers.get('Content-Length', 0))
                    url_candidates.append((content_length, version_url))
                except (BadResponseException, KeyError, IndexError) as err:
                    self._context.error(f"Video URL candidate {idx+1}/{len(version_urls)} for {self}: {err}")
            if not url_candidates:
                return version_urls[0]
            url_candidates.sort()
            return url_candidates[-1][1]
        return None

class Story:
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any]) -> None:
        self._context: InstaloaderContext = context
        self._node: Dict[str, Any] = node
        self._unique_id: Optional[str] = None
        self._owner_profile: Optional[Profile] = None
        self._iphone_struct_: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return '<Story by {} changed {:%Y-%m-%d_%H-%M-%S_UTC}>'.format(self.owner_username, self.latest_media_utc)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Story):
            return self.unique_id == o.unique_id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.unique_id)

    @property
    def unique_id(self) -> Union[str, int]:
        if not self._unique_id:
            id_list: List[int] = [item.mediaid for item in self.get_items()]
            id_list.sort()
            self._unique_id = ''.join([str(self.owner_id)] + list(map(str, id_list)))
        return self._unique_id

    @property
    def last_seen_local(self) -> Optional[datetime]:
        if self._node['seen']:
            return datetime.fromtimestamp(self._node['seen'])
        return None

    @property
    def last_seen_utc(self) -> Optional[datetime]:
        if self._node['seen']:
            return datetime.utcfromtimestamp(self._node['seen'])
        return None

    @property
    def latest_media_local(self) -> datetime:
        return datetime.fromtimestamp(self._node['latest_reel_media'])

    @property
    def latest_media_utc(self) -> datetime:
        return datetime.utcfromtimestamp(self._node['latest_reel_media'])

    @property
    def itemcount(self) -> int:
        return len(self._node['items'])

    @property
    def owner_profile(self) -> Profile:
        if not self._owner_profile:
            self._owner_profile = Profile(self._context, self._node['user'])
        return self._owner_profile

    @property
    def owner_username(self) -> str:
        return self.owner_profile.username

    @property
    def owner_id(self) -> int:
        return self.owner_profile.userid

    def _fetch_iphone_struct(self) -> None:
        if self._context.iphone_support and self._context.is_logged_in and not self._iphone_struct_:
            data: Dict[str, Any] = self._context.get_iphone_json(path=f'api/v1/feed/reels_media/?reel_ids={self.owner_id}', params={})
            self._iphone_struct_ = data['reels'][str(self.owner_id)]

    def get_items(self) -> Iterator[StoryItem]:
        self._fetch_iphone_struct()
        for item in reversed(self._node['items']):
            if self._iphone_struct_ is not None:
                for iphone_struct_item in self._iphone_struct_['items']:
                    if iphone_struct_item['pk'] == int(item['id']):
                        item['iphone_struct'] = iphone_struct_item
                        break
            yield StoryItem(self._context, item, self.owner_profile)

class Highlight(Story):
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any], owner: Optional[Profile] = None) -> None:
        super().__init__(context, node)
        self._owner_profile: Optional[Profile] = owner
        self._items: Optional[List[Dict[str, Any]]] = None
        self._iphone_struct_: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return '<Highlight by {}: {}>'.format(self.owner_username, self.title)

    @property
    def unique_id(self) -> int:
        return int(self._node['id'])

    @property
    def owner_profile(self) -> Profile:
        if not self._owner_profile:
            self._owner_profile = Profile(self._context, self._node['owner'])
        return self._owner_profile

    @property
    def title(self) -> str:
        return self._node['title']

    @property
    def cover_url(self) -> str:
        return self._node['cover_media']['thumbnail_src']

    @property
    def cover_cropped_url(self) -> str:
        return self._node['cover_media_cropped_thumbnail']['url']

    def _fetch_items(self) -> None:
        if not self._items:
            self._items = self._context.graphql_query("45246d3fe16ccc6577e0bd297a5db1ab",
                                                      {"reel_ids": [], "tag_names": [], "location_ids": [],
                                                       "highlight_reel_ids": [str(self.unique_id)],
                                                       "precomposed_overlay": False})['data']['reels_media'][0]['items']

    def _fetch_iphone_struct(self) -> None:
        if self._context.iphone_support and self._context.is_logged_in and not self._iphone_struct_:
            data: Dict[str, Any] = self._context.get_iphone_json(path=f'api/v1/feed/reels_media/?reel_ids=highlight:{self.unique_id}', params={})
            self._iphone_struct_ = data['reels'][f'highlight:{self.unique_id}']

    @property
    def itemcount(self) -> int:
        self._fetch_items()
        assert self._items is not None
        return len(self._items)

    def get_items(self) -> Iterator[StoryItem]:
        self._fetch_items()
        self._fetch_iphone_struct()
        assert self._items is not None
        for item in self._items:
            if self._iphone_struct_ is not None:
                for iphone_struct_item in self._iphone_struct_['items']:
                    if iphone_struct_item['pk'] == int(item['id']):
                        item['iphone_struct'] = iphone_struct_item
                        break
            yield StoryItem(self._context, item, self.owner_profile)

class Hashtag:
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any]) -> None:
        assert "name" in node
        self._context: InstaloaderContext = context
        self._node: Dict[str, Any] = node
        self._has_full_metadata: bool = False

    @classmethod
    def from_name(cls, context: InstaloaderContext, name: str) -> "Hashtag":
        hashtag = cls(context, {'name': name.lower()})
        hashtag._obtain_metadata()
        return hashtag

    @property
    def name(self) -> str:
        return self._node["name"].lower()

    def _query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        json_response: Dict[str, Any] = self._context.get_iphone_json("api/v1/tags/web_info/", {**params, "tag_name": self.name})
        return json_response["graphql"]["hashtag"] if "graphql" in json_response else json_response["data"]

    def _obtain_metadata(self) -> None:
        if not self._has_full_metadata:
            self._node = self._query({"__a": 1, "__d": "dis"})
            self._has_full_metadata = True

    def _asdict(self) -> Dict[str, Any]:
        json_node: Dict[str, Any] = self._node.copy()
        json_node.pop("edge_hashtag_to_top_posts", None)
        json_node.pop("top", None)
        json_node.pop("edge_hashtag_to_media", None)
        json_node.pop("recent", None)
        return json_node

    def __repr__(self) -> str:
        return f"<Hashtag #{self.name}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Hashtag):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def _metadata(self, *keys: str) -> Any:
        try:
            d: Any = self._node
            for key in keys:
                d = d[key]
            return d
        except KeyError:
            self._obtain_metadata()
            d = self._node
            for key in keys:
                d = d[key]
            return d

    @property
    def hashtagid(self) -> int:
        return int(self._metadata("id"))

    @property
    def profile_pic_url(self) -> str:
        return self._metadata("profile_pic_url")

    @property
    def description(self) -> Optional[str]:
        return self._metadata("description")

    @property
    def allow_following(self) -> bool:
        return bool(self._metadata("allow_following"))

    @property
    def is_following(self) -> bool:
        try:
            return self._metadata("is_following")
        except KeyError:
            return bool(self._metadata("following"))

    def get_top_posts(self) -> Iterator[Post]:
        try:
            yield from (Post(self._context, edge["node"])
                        for edge in self._metadata("edge_hashtag_to_top_posts", "edges"))
        except KeyError:
            yield from SectionIterator(
                self._context,
                lambda d: d["data"]["top"],
                lambda m: Post.from_iphone_struct(self._context, m),
                f"explore/tags/{self.name}/",
                self._metadata("top"),
            )

    @property
    def mediacount(self) -> int:
        try:
            return self._metadata("edge_hashtag_to_media", "count")
        except KeyError:
            return self._metadata("media_count")

    def get_posts(self) -> Iterator[Post]:
        try:
            self._metadata("edge_hashtag_to_media", "edges")
            self._metadata("edge_hashtag_to_media", "page_info")
            conn: Dict[str, Any] = self._metadata("edge_hashtag_to_media")
            yield from (Post(self._context, edge["node"]) for edge in conn["edges"])
            while conn["page_info"]["has_next_page"]:
                data: Dict[str, Any] = self._query({'__a': 1, 'max_id': conn["page_info"]["end_cursor"]})
                conn = data["edge_hashtag_to_media"]
                yield from (Post(self._context, edge["node"]) for edge in conn["edges"])
        except KeyError:
            yield from SectionIterator(
                self._context,
                lambda d: d["data"]["recent"],
                lambda m: Post.from_iphone_struct(self._context, m),
                f"explore/tags/{self.name}/",
                self._metadata("recent"),
            )

    def get_all_posts(self) -> Iterator[Post]:
        sorted_top_posts = iter(sorted(islice(self.get_top_posts(), 9), key=lambda p: p.date_utc, reverse=True))
        other_posts = self.get_posts_resumable()
        next_top = next(sorted_top_posts, None)
        next_other = next(other_posts, None)
        while next_top is not None or next_other is not None:
            if next_other is None:
                assert next_top is not None
                yield next_top
                yield from sorted_top_posts
                break
            if next_top is None:
                assert next_other is not None
                yield next_other
                yield from other_posts
                break
            if next_top == next_other:
                yield next_top
                next_top = next(sorted_top_posts, None)
                next_other = next(other_posts, None)
                continue
            if next_top.date_utc > next_other.date_utc:
                yield next_top
                next_top = next(sorted_top_posts, None)
            else:
                yield next_other
                next_other = next(other_posts, None)

    def get_posts_resumable(self) -> NodeIterator[Post]:
        return NodeIterator(
            self._context, "9b498c08113f1e09617a1703c22b2f32",
            lambda d: d['data']['hashtag']['edge_hashtag_to_media'],
            lambda n: Post(self._context, n),
            {'tag_name': self.name},
            f"https://www.instagram.com/explore/tags/{self.name}/"
        )

class TopSearchResults:
    def __init__(self, context: InstaloaderContext, searchstring: str) -> None:
        self._context: InstaloaderContext = context
        self._searchstring: str = searchstring
        self._node: Dict[str, Any] = context.get_json('web/search/topsearch/',
                                                       params={'context': 'blended',
                                                               'query': searchstring,
                                                               'include_reel': False,
                                                               '__a': 1})

    def get_profiles(self) -> Iterator[Profile]:
        for user in self._node.get('users', []):
            user_node = user['user']
            if 'pk' in user_node:
                user_node['id'] = user_node['pk']
            yield Profile(self._context, user_node)

    def get_prefixed_usernames(self) -> Iterator[str]:
        for user in self._node.get('users', []):
            username: str = user.get('user', {}).get('username', '')
            if username.startswith(self._searchstring):
                yield username

    def get_locations(self) -> Iterator[PostLocation]:
        for location in self._node.get('places', []):
            place: Dict[str, Any] = location.get('place', {})
            slug: Optional[str] = place.get('slug')
            loc: Dict[str, Any] = place.get('location', {})
            yield PostLocation(int(loc['pk']), loc['name'], slug, None, loc.get('lat'), loc.get('lng'))

    def get_hashtag_strings(self) -> Iterator[str]:
        for hashtag in self._node.get('hashtags', []):
            name: Optional[str] = hashtag.get('hashtag', {}).get('name')
            if name:
                yield name

    def get_hashtags(self) -> Iterator[Hashtag]:
        for hashtag in self._node.get('hashtags', []):
            node: Dict[str, Any] = hashtag.get('hashtag', {})
            if 'name' in node:
                yield Hashtag(self._context, node)

    @property
    def searchstring(self) -> str:
        return self._searchstring

class TitlePic:
    def __init__(self, profile: Optional[Profile], target: Union[str, Path], typename: str,
                 filename: str, date_utc: Optional[datetime]) -> None:
        self._profile: Optional[Profile] = profile
        self._target: Union[str, Path] = target
        self._typename: str = typename
        self._filename: str = filename
        self._date_utc: Optional[datetime] = date_utc

    @property
    def profile(self) -> Union[str, Path]:
        return self._profile.username.lower() if self._profile is not None else self._target

    @property
    def owner_username(self) -> Union[str, Path]:
        return self.profile

    @property
    def owner_id(self) -> Union[str, Path]:
        return str(self._profile.userid) if self._profile is not None else self._target

    @property
    def target(self) -> Union[str, Path]:
        return self._target

    @property
    def typename(self) -> str:
        return self._typename

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def date_utc(self) -> Optional[datetime]:
        return self._date_utc

    @property
    def date(self) -> Optional[datetime]:
        return self.date_utc

    @property
    def date_local(self) -> Optional[datetime]:
        return self._date_utc.astimezone() if self._date_utc is not None else None

JsonExportable = Union[Post, Profile, StoryItem, Hashtag, FrozenNodeIterator]

def get_json_structure(structure: JsonExportable) -> dict:
    return {
        'node': structure._asdict(),
        'instaloader': {'version': __version__, 'node_type': structure.__class__.__name__}
    }

def save_structure_to_file(structure: JsonExportable, filename: str) -> None:
    json_structure: dict = get_json_structure(structure)
    compress: bool = filename.endswith('.xz')
    if compress:
        with lzma.open(filename, 'wt', check=lzma.CHECK_NONE) as fp:
            json.dump(json_structure, fp=fp, separators=(',', ':'))
    else:
        with open(filename, 'wt') as fp:
            json.dump(json_structure, fp=fp, indent=4, sort_keys=True)

def load_structure(context: InstaloaderContext, json_structure: dict) -> JsonExportable:
    if 'node' in json_structure and 'instaloader' in json_structure and 'node_type' in json_structure['instaloader']:
        node_type: str = json_structure['instaloader']['node_type']
        if node_type == "Post":
            return Post(context, json_structure['node'])
        elif node_type == "Profile":
            return Profile(context, json_structure['node'])
        elif node_type == "StoryItem":
            return StoryItem(context, json_structure['node'])
        elif node_type == "Hashtag":
            return Hashtag(context, json_structure['node'])
        elif node_type == "FrozenNodeIterator":
            if 'first_node' not in json_structure['node']:
                json_structure['node']['first_node'] = None
            return FrozenNodeIterator(**json_structure['node'])
    elif 'shortcode' in json_structure:
        return Post.from_shortcode(context, json_structure['shortcode'])
    raise InvalidArgumentException("Passed json structure is not an Instaloader JSON")

def load_structure_from_file(context: InstaloaderContext, filename: str) -> JsonExportable:
    compressed: bool = filename.endswith('.xz')
    if compressed:
        fp = lzma.open(filename, 'rt')
    else:
        fp = open(filename, 'rt')
    json_structure: dict = json.load(fp)
    fp.close()
    return load_structure(context, json_structure)