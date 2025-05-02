from typing import Any, Callable, Dict, Generic, List, Optional, Pattern, Set, Tuple, TypeVar, Union, cast
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from functools import lru_cache
from re import Match
from urllib.parse import parse_qs, quote, urljoin, urlsplit, urlunsplit
from xml.etree.ElementTree import Element, SubElement

import ahocorasick
import dateutil.parser
import dateutil.tz
import lxml.etree
import markdown
import markdown.blockprocessors
import markdown.inlinepatterns
import markdown.postprocessors
import markdown.preprocessors
import markdown.treeprocessors
import markdown.util
import re2
import regex
import requests
import uri_template
import urllib3.exceptions
from django.conf import settings
from markdown.blockparser import BlockParser
from markdown.extensions import codehilite, nl2br, sane_lists, tables
from tlds import tld_set
from typing_extensions import Self, override

from zerver.lib import mention
from zerver.lib.cache import cache_with_key
from zerver.lib.camo import get_camo_url
from zerver.lib.emoji import EMOTICON_RE, codepoint_to_name, name_to_codepoint, translate_emoticons
from zerver.lib.emoji_utils import emoji_to_hex_codepoint, unqualify_emoji
from zerver.lib.exceptions import MarkdownRenderingError
from zerver.lib.markdown import fenced_code
from zerver.lib.markdown.fenced_code import FENCE_RE
from zerver.lib.mention import (
    BEFORE_MENTION_ALLOWED_REGEX,
    ChannelTopicInfo,
    FullNameInfo,
    MentionBackend,
    MentionData,
    get_user_group_mention_display_name,
)
from zerver.lib.outgoing_http import OutgoingSession
from zerver.lib.subdomains import is_static_or_current_realm_url
from zerver.lib.tex import render_tex
from zerver.lib.thumbnail import (
    MarkdownImageMetadata,
    get_user_upload_previews,
    rewrite_thumbnailed_images,
)
from zerver.lib.timeout import unsafe_timeout
from zerver.lib.timezone import common_timezones
from zerver.lib.types import LinkifierDict
from zerver.lib.url_encoding import encode_stream, hash_util_encode
from zerver.lib.url_preview.types import UrlEmbedData, UrlOEmbedData
from zerver.models import Message, Realm, UserProfile
from zerver.models.linkifiers import linkifiers_for_realm
from zerver.models.realm_emoji import EmojiInfo, get_name_keyed_dict_for_active_realm_emoji

ReturnT = TypeVar("ReturnT")

html_safelisted_schemes = (
    "bitcoin",
    "geo",
    "im",
    "irc",
    "ircs",
    "magnet",
    "mailto",
    "matrix",
    "mms",
    "news",
    "nntp",
    "openpgp4fpr",
    "sip",
    "sms",
    "smsto",
    "ssh",
    "tel",
    "urn",
    "webcal",
    "wtai",
    "xmpp",
)
allowed_schemes = ("http", "https", "ftp", "file", *html_safelisted_schemes)

class LinkInfo(TypedDict):
    parent: Element
    title: Optional[str]
    index: Optional[int]
    remove: Optional[Element]

@dataclass
class MessageRenderingResult:
    rendered_content: str
    mentions_topic_wildcard: bool
    mentions_stream_wildcard: bool
    mentions_user_ids: Set[int]
    mentions_user_group_ids: Set[int]
    alert_words: Set[str]
    links_for_preview: Set[str]
    user_ids_with_alert_words: Set[int]
    potential_attachment_path_ids: List[str]
    thumbnail_spinners: Set[str]

@dataclass
class DbData:
    mention_data: MentionData
    realm_url: str
    realm_alert_words_automaton: Optional[ahocorasick.Automaton]
    active_realm_emoji: Dict[str, EmojiInfo]
    sent_by_bot: bool
    stream_names: Dict[str, int]
    topic_info: Dict[ChannelTopicInfo, Optional[int]]
    translate_emoticons: bool
    user_upload_previews: Dict[str, MarkdownImageMetadata]

version = 1

_T = TypeVar("_T")
ElementStringNone: TypeAlias = Union[Element, str, None]

EMOJI_REGEX = r"(?P<syntax>:[\w\-\+]+:)"

def verbose_compile(pattern: str) -> Pattern[str]:
    return re.compile(
        rf"^(.*?){pattern}(.*?)$",
        re.DOTALL | re.VERBOSE,
    )

STREAM_LINK_REGEX = rf"""
                     {BEFORE_MENTION_ALLOWED_REGEX} # Start after whitespace or specified chars
                     \#\*\*                         # and after hash sign followed by double asterisks
                         (?P<stream_name>[^\*]+)    # stream name can contain anything
                     \*\*                           # ends by double asterisks
                    """

@lru_cache(None)
def get_compiled_stream_link_regex() -> Pattern[str]:
    return re.compile(
        STREAM_LINK_REGEX,
        re.DOTALL | re.VERBOSE,
    )

STREAM_TOPIC_LINK_REGEX = rf"""
                     {BEFORE_MENTION_ALLOWED_REGEX}  # Start after whitespace or specified chars
                     \#\*\*                          # and after hash sign followed by double asterisks
                         (?P<stream_name>[^\*>]+)    # stream name can contain anything except >
                         >                           # > acts as separator
                         (?P<topic_name>[^\*]*)      # topic name can be an empty string or contain anything
                     \*\*                            # ends by double asterisks
                   """

@lru_cache(None)
def get_compiled_stream_topic_link_regex() -> Pattern[str]:
    return re.compile(
        STREAM_TOPIC_LINK_REGEX,
        re.DOTALL | re.VERBOSE,
    )

STREAM_TOPIC_MESSAGE_LINK_REGEX = rf"""
                     {BEFORE_MENTION_ALLOWED_REGEX}  # Start after whitespace or specified chars
                     \#\*\*                          # and after hash sign followed by double asterisks
                         (?P<stream_name>[^\*>]+)    # stream name can contain anything except >
                         >                           # > acts as separator
                         (?P<topic_name>[^\*]*)      # topic name can be an empty string or contain anything
                         @
                         (?P<message_id>\d+)         # message id
                     \*\*                            # ends by double asterisks
                   """

@lru_cache(None)
def get_compiled_stream_topic_message_link_regex() -> Pattern[str]:
    return re.compile(
        STREAM_TOPIC_MESSAGE_LINK_REGEX,
        re.DOTALL | re.VERBOSE,
    )

@lru_cache(None)
def get_web_link_regex() -> Pattern[str]:
    tlds = r"|".join(list_of_tlds())
    inner_paren_contents = r"[^\s()\"]*"
    paren_group = r"""
                    [^\s()\"]*?            # Containing characters that won't end the URL
                    (?: \( %s \)           # and more characters in matched parens
                        [^\s()\"]*?        # followed by more characters
                    )*                     # zero-or-more sets of paired parens
                   """
    nested_paren_chunk = paren_group
    for i in range(6):
        nested_paren_chunk %= (paren_group,)
    nested_paren_chunk %= (inner_paren_contents,)

    file_links = r"| (?:file://(/[^/ ]*)+/?)" if settings.ENABLE_FILE_LINKS else r""
    REGEX = rf"""
        (?<![^\s'"\(,:<])    # Start after whitespace or specified chars
                             # (Double-negative lookbehind to allow start-of-string)
        (?P<url>             # Main group
            (?:(?:           # Domain part
                https?://[\w.:@-]+?   # If it has a protocol, anything goes.
               |(?:                   # Or, if not, be more strict to avoid false-positives
                    (?:[\w-]+\.)+     # One or more domain components, separated by dots
                    (?:{tlds})        # TLDs
                )
            )
            (?:/             # A path, beginning with /
                {nested_paren_chunk}           # zero-to-6 sets of paired parens
            )?)              # Path is optional
            | (?:[\w.-]+\@[\w.-]+\.[\w]+) # Email is separate, since it can't have a path
            {file_links}               # File path start with file:///, enable by setting ENABLE_FILE_LINKS=True
            | (?:bitcoin:[13][a-km-zA-HJ-NP-Z1-9]{{25,34}})  # Bitcoin address pattern, see https://mokagio.github.io/tech-journal/2014/11/21/regex-bitcoin.html
        )
        (?=                            # URL must be followed by (not included in group)
            [!:;\?\),\.\'\"\>]*         # Optional punctuation characters
            (?:\Z|\s)                  # followed by whitespace or end of string
        )
        """
    return verbose_compile(REGEX)

def clear_web_link_regex_for_testing() -> None:
    get_web_link_regex.cache_clear()

markdown_logger = logging.getLogger()

def rewrite_local_links_to_relative(db_data: Optional[DbData], link: str) -> str:
    if db_data:
        realm_url_prefix = db_data.realm_url + "/"
        if link.startswith((realm_url_prefix + "#", realm_url_prefix + "user_uploads/")):
            return link.removeprefix(realm_url_prefix)
    return link

def url_embed_preview_enabled(
    message: Optional[Message] = None, realm: Optional[Realm] = None, no_previews: bool = False
) -> bool:
    if not settings.INLINE_URL_EMBED_PREVIEW:
        return False
    if no_previews:
        return False
    if realm is None and message is not None:
        realm = message.get_realm()
    if realm is None:
        return True
    return realm.inline_url_embed_preview

def image_preview_enabled(
    message: Optional[Message] = None, realm: Optional[Realm] = None, no_previews: bool = False
) -> bool:
    if not settings.INLINE_IMAGE_PREVIEW:
        return False
    if no_previews:
        return False
    if realm is None and message is not None:
        realm = message.get_realm()
    if realm is None:
        return True
    return realm.inline_image_preview

def list_of_tlds() -> List[str]:
    common_false_positives = {"java", "md", "mov", "py", "zip"}
    return sorted(tld_set - common_false_positives, key=len, reverse=True)

def walk_tree(
    root: Element, processor: Callable[[Element], Optional[_T]], stop_after_first: bool = False
) -> List[_T]:
    results = []
    queue = deque([root])
    while queue:
        currElement = queue.popleft()
        for child in currElement:
            queue.append(child)
            result = processor(child)
            if result is not None:
                results.append(result)
                if stop_after_first:
                    return results
    return results

@dataclass
class ElementFamily:
    grandparent: Optional[Element]
    parent: Element
    child: Element
    in_blockquote: bool

T = TypeVar("T")

class ResultWithFamily(Generic[T]):
    family: ElementFamily
    result: T

    def __init__(self, family: ElementFamily, result: T) -> None:
        self.family = family
        self.result = result

class ElementPair:
    parent: Optional["ElementPair"]
    value: Element

    def __init__(self, parent: Optional["ElementPair"], value: Element) -> None:
        self.parent = parent
        self.value = value

def walk_tree_with_family(
    root: Element,
    processor: Callable[[Element], Optional[_T]],
) -> List[ResultWithFamily[_T]]:
    results = []
    queue = deque([ElementPair(parent=None, value=root)])
    while queue:
        currElementPair = queue.popleft()
        for child in currElementPair.value:
            queue.append(ElementPair(parent=currElementPair, value=child))
            result = processor(child)
            if result is not None:
                if currElementPair.parent is not None:
                    grandparent_element = currElementPair.parent
                    grandparent: Optional[Element] = grandparent_element.value
                else:
                    grandparent = None
                family = ElementFamily(
                    grandparent=grandparent,
                    parent=currElementPair.value,
                    child=child,
                    in_blockquote=has_blockquote_ancestor(currElementPair),
                )
                results.append(
                    ResultWithFamily(
                        family=family,
                        result=result,
                    )
                )
    return results

def has_blockquote_ancestor(element_pair: Optional[ElementPair]) -> bool:
    if element_pair is None:
        return False
    elif element_pair.value.tag == "blockquote":
        return True
    else:
        return has_blockquote_ancestor(element_pair.parent)

@cache_with_key(lambda tweet_id: tweet_id, cache_name="database")
def fetch_tweet_data(tweet_id: str) -> Optional[Dict[str, Any]]:
    raise NotImplementedError("Twitter desupported their v1 API")

class OpenGraphSession(OutgoingSession):
    def __init__(self) -> None:
        super().__init__(role="markdown", timeout=1)

def fetch_open_graph_image(url: str) -> Optional[Dict[str, Any]]:
    og: Dict[str, Optional[str]] = {"image": None, "title": None, "desc": None}
    try:
        with OpenGraphSession().get(
            url, headers={"Accept": "text/html,application/xhtml+xml"}, stream=True
        ) as res:
            if res.status_code != requests.codes.ok:
                return None
            m = EmailMessage()
            m["Content-Type"] = res.headers.get("Content-Type")
            mimetype = m.get_content_type()
            if mimetype not in ("text/html", "application/xhtml+xml"):
                return None
            html = mimetype == "text/html"
            res.raw.decode_content = True
            for event, element in lxml.etree.iterparse(
                res.raw, events=("start",), no_network=True, remove_comments=True, html=html
            ):
                parent = element.getparent()
                if parent is not None:
                    parent.text = None
                    parent.remove(element)
                if element.tag in ("body", "{http://www.w3.org/1999/xhtml}body"):
                    break
                elif element.tag in ("meta", "{http://www.w3.org/1999/xhtml}meta"):
                    if element.get("property") == "og:image":
                        content = element.get("content")
                        if content is not None:
                            og["image"] = urljoin(res.url, content)
                    elif element.get("property") == "og:title":
                        og["title"] = element.get("content")
                    elif element.get("property") == "og:description":
                        og["desc"] = element.get("content")
    except (requests.RequestException, urllib3.exceptions.HTTPError):
        return None
    return None if og["image"] is None else og

def get_tweet_id(url: str) -> Optional[str]:
    parsed_url = urlsplit(url)
    if not (parsed_url.netloc == "twitter.com" or parsed_url.netloc.endswith(".twitter.com")):
        return None
    to_match = parsed_url.path
    if parsed_url.path == "/" and len(parsed_url.fragment) > 5:
        to_match = parsed_url.fragment
    tweet_id_match = re.match(
        r"^!?/.*?/status(es)?/(?P<tweetid>\d{10,30})(/photo/[0-9])?/?$", to_match
    )
    if not tweet_id_match:
        return None
    return tweet_id_match.group("tweetid")

class InlineImageProcessor(markdown.treeprocessors.Treeprocessor):
    def __init__(self, zmd: "ZulipMarkdown") -> None:
        super().__init__(zmd)
        self.zmd = zmd

    @override
    def run(self, root: Element) -> None:
        found_imgs = walk_tree(root, lambda e: e if e.tag == "img" else None)
        for img in found_imgs:
            url = img.get("src")
            assert url is not None
            if is_static_or_current_realm_url(url, self.zmd.zulip_realm):
                continue
            img.set("src", get_camo_url(url))

class InlineVideoProcessor(markdown.treeprocessors.Treeprocessor):
    def __init__(self, zmd: "ZulipMarkdown") -> None:
        super().__init__(zmd)
        self.zmd = zmd

    @override
    def run(self, root: Element) -> None:
        found_videos = walk_tree(root, lambda e: e if e.tag == "video" else None)
        for video in found_videos:
            url = video.get("src")
            assert url is not None
            if is_static_or_current_realm_url(url, self.zmd.zulip_realm):
                continue
            video.set("src", get_camo_url(url))

class BacktickInlineProcessor(markdown.inlinepatterns.BacktickInlineProcessor):
    @override
    def handleMatch(  # type: ignore[override] # https://github.com/python/mypy/issues/10197
        self, m: Match[str], data: str
    ) -> Tuple[Union[Element, str, None], Optional[int], Optional[int]]:
        el, start, end = ret = super().handleMatch(m, data)
        if el is not None and m.group(3):
            assert isinstance(el, Element)
            el.text = markdown.util.AtomicString(markdown.util.code_escape(m.group(3)))
        return ret

IMAGE_EXTENSIONS = [".bmp", ".gif", ".jpe", ".jpeg", ".jpg", ".png", ".webp"]

class InlineInterestingLinkProcessor(markdown.treeprocessors.Treeprocessor):
    TWITTER_MAX_IMAGE_HEIGHT = 400
    TWITTER_MAX_TO_PREVIEW = 3
    INLINE_PREVIEW_LIMIT_PER_MESSAGE = 24

    def __init__(self, zmd: "ZulipMarkdown") -> None:
        super().__init__(zmd)
        self.zmd = zmd

    def add_a(
        self,
        root: Element,
        image_url: str,
        link: str,
        title: Optional[str] = None,
        desc: Optional[str] = None,
        class_attr: str = "message_inline_image",
        data_id: Optional[str] = None,
        insertion_index: Optional[int] = None,
        already_thumbnailed: bool = False,
    ) -> None:
        desc = desc if desc is not None else ""
        if "message_inline_image" in class_attr and self.zmd.zulip_message:
            self.zmd.zulip_message.has_image = True
        if insertion_index is not None:
            div = Element("div")
            root.insert(insertion_index, div)
        else:
            div = SubElement(root, "div")
        div.set("class", class_attr)
        a = SubElement(div, "a")
        a.set("href", link)
        if title is not None:
            a.set("title", title)
        if data_id is not None:
            a.set("data-id", data_id)
        img = SubElement(a, "img")
        if image_url.startswith("/user_uploads/") and self.zmd.zulip_db_data:
            path_id = image_url.removeprefix("/user_uploads/")
            assert path_id in self.zmd.zulip_db_data.user_upload_previews
            metadata = self.zmd.zulip_db_data.user_upload_previews[path_id]
            img.set("class", "image-loading-placeholder")
            img.set("src", "/static/images/loading/loader-black.svg")
            img.set(
                "data-original-dimensions",
                f"{metadata.original_width_px}x{metadata.original_height_px}",
            )
            if metadata.original_content_type:
                img.set(
                    "data-original-content-type",
                    metadata.original_content_type,
                )
        else:
            img.set("src", image_url)
        if class_attr == "message_inline_ref":
            summary_div = SubElement(div, "div")
            title_div = SubElement(summary_div, "div")
            title_div.set("class", "message_inline_image_title")
            title_div.text = title
            desc_div = SubElement(summary_div, "desc")
            desc_div.set("class", "message_inline_image_desc")

    def add_oembed_data(self, root: Element, link: str, extracted_data: UrlOEmbedData) -> None:
        if extracted_data.image is None:
            return
        if extracted_data.type == "photo":
            self.add_a(
                root,
                image_url=extracted_data.image,
                link=link,
                title=extracted_data.title,
            )
        elif extracted_data.type == "video":
            self.add_a(
                root,
                image_url=extracted_data.image,
                link=link,
                title=extracted_data.title,
                desc=extracted_data.description,
                class_attr="embed-video message_inline_image",
                data_id=extracted_data.html,
                already_thumbnailed=True,
            )

    def add_embed(self, root: Element, link: str, extracted_data: UrlEmbedData) -> None:
        if isinstance(extracted_data, UrlOEmbedData):
            self.add_oembed_data(root, link, extracted_data)
            return
        if extracted_data.image is None:
            return
        container = SubElement(root, "div")
        container.set("class", "message_embed")
        img_link = get_camo_url(extracted_data.image)
        img = SubElement(container, "a")
        img.set(
            "style",
            'background-image: url("'
            + img_link.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\a ")
            + '")',
        )
        img.set("href", link)
        img.set("class", "message_embed_image")
        data_container = SubElement(container, "div")
        data_container.set("class", "data-container")
        if extracted_data.title:
            title_elm = SubElement(data_container, "div")
            title_elm.set("class", "message_embed_title")
            a = SubElement(title_elm, "a")
            a.set("href", link)
            a.set("title", extracted_data.title)
            a.text = extracted_data.title
        if extracted_data.description:
            description_elm = SubElement(data_container, "div")
            description_elm.set("class", "message_embed_description")
            description_elm.text = extracted_data.description

    def get_actual_image_url(self, url: str) -> str:
        parsed_url = urlsplit(url)
        if parsed_url.netloc == "github.com" or parsed_url.netloc.endswith(".github.com"):
            split_path = parsed_url.path.split("/")
            if len(split_path) > 3 and split_path[3] == "blob":
                return urljoin(
                    "https://raw.githubusercontent.com", "/".join(split_path[0:3] + split_path[4:])
        return url

    def is_image(self, url: str) -> bool:
        if not self.zmd.image_preview_enabled:
            return False
        parsed_url = urlsplit(url)
        if parsed_url.netloc == "pasteboard.co":
            return False
        if url.startswith("/user_uploads/") and self.zmd.zulip_db_data:
            path_id = url.removeprefix("/user_uploads/")
            return path_id in self.zmd.zulip_db_data.user_upload_previews
        return any(parsed_url.path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

    def corrected_image_source(self, url: str) -> Optional[str]:
        parsed_url = urlsplit(url)
        if parsed_url.netloc.lower().endswith(".wikipedia.org") and parsed_url.path.startswith(
            "/wiki/File:"
        ):
            newpath = parsed_url.path.replace("/wiki/File:", "/wiki/Special:FilePath/File:", 1)
            return parsed_url._replace(path=newpath).geturl()
        if parsed_url.netloc == "linx.li":
            return "https://linx.li/s" + parsed_url.path
        return None

    def dropbox_image(self, url: str) -> Optional[Dict[str, Any]]:
        parsed_url = urlsplit(url)
        if parsed_url.netloc == "dropbox.com" or parsed_url.netloc.endswith(".dropbox.com"):
            is_album = parsed_url.path.startswith("/sc/") or parsed_url.path.startswith("/photos/")
            if not (
                parsed_url.path.startswith("/s/") or parsed_url.path.startswith("/sh/") or is_album
            ):
                return None
            image_info = fetch_open_graph_image(url)
            is_image = is_album or self.is_image(url)
            if is_album or not is_image:
                if image_info is None:
                    return None
                image_info["is_image"] = is_image
                return image_info
            if image_info is None:
                image_info = {}
            image_info["is_image"] = True
            image_info["image"] = parsed_url._replace(query="raw=1").geturl()
            return image_info
        return None

    def youtube_id(self, url: str) -> Optional[str]:
        if not self.zmd.image_preview_enabled:
            return None
        id = None
        split_url = urlsplit(url)
        if split_url.scheme in ("http", "https"):
            if split_url.hostname in (
                "m.youtube.com",
                "www.youtube.com",
                "www.youtube-nocookie.com",
                "youtube.com",
                "youtube-nocookie.com",
            ):
                query = parse_qs(split_url.query)
                if split_url.path in ("/watch", "/watch_popup") and "v" in query:
                    id = query["v"][0]
                elif split_url.path == "/watch_videos" and "video_ids" in query:
                    id = query["video_ids"][0].split(",", 1)[0]
                elif split_url.path.startswith(("/embed/", "/shorts/", "/v/")):
                    id = split_url.path.split("/", 3)[2]
            elif split_url.hostname == "youtu.be" and split_url.path.startswith("/"):
                id = split_url.path.removeprefix("/")
        if id is not None and re.fullmatch(r"[0-9A-Za-z_-]+", id):
            return id
        return None

    def youtube_title(self, extracted_data: UrlEmbedData) -> Optional[str]:
        if extracted_data.title is not None:
            return f"YouTube - {extracted_data.title}"
        return None

    def youtube_image(self, url: str) -> Optional[str]:
        yt_id = self.youtube_id(url)
        if yt_id is not None:
            return f"https://i.ytimg.com/vi/{yt_id}/default.jpg"
        return None

    def vimeo_id(self, url: str) -> Optional[str]:
        if not self.zmd.image_preview_enabled:
            return None
        vimeo_re = (
            r"^((http|https)?:\/\/(www\.)?vimeo.com\/"
            r"(?:channels\/(?:\w+\/)?|groups\/"
            r"([^\/]*)\/videos\/|)(\d+)(?:|\/\?))$"
        )
        match = re.match(vimeo_re, url)
        if match is None:
            return None
        return match.group(5)

    def vimeo_title(self, extracted_data: UrlEmbedData) -> Optional[str]:
        if extracted_data.title is not None:
            return f"Vimeo - {extracted_data.title}"
        return None

    def twitter_text(
        self,
        text: str,
        urls: List[Dict[str, str]],
        user_mentions: List[Dict[str, Any]]],
        media: List[Dict[str, Any]]],
    ) -> Element:
        to_process: List[Dict[str, Any]] = []
        for url_data in urls:
            to_process.extend(
                {
                    "type": "url",
                    "start": match.start(),
                    "end": match.end(),
                    "url": url_data["url"],
                    "text": url_data["expanded_url"],
                }
                for match in re.finditer(re.escape(url_data["url"]), text, re.IGNORECASE)
            )
        for user_mention in user_mentions:
            screen_name = user_mention["screen_name"]
            mention_string = "@" + screen_name
            to_process.extend(
                {
                    "type": "mention",
                    "start": match.start(),
                    "end": match.end(),
                    "url": "https://twitter.com/" + quote(screen_name),
                    "text": mention_string,
                }
                for match in re.finditer(re.escape(mention_string), text, re.IGNORECASE)
            )
        for media_item in media:
            short_url = media_item["url"]
            expanded_url = media_item["expanded_url"]
            to_process.extend(
                {
                    "type": "media",
                    "start": match.start(),
                    "end": match.end(),
                    "url": short_url,
                    "text": expanded_url,
                }
                for match in re.finditer(re.escape(short_url), text, re.IGNORECASE)
            )
        for match in POSSIBLE_EMOJI_RE.finditer(text):
            orig_syntax = match.group("syntax")
            codepoint = emoji_to_hex_codepoint(unqualify_emoji(orig_syntax))
            if codepoint in codepoint_to_name:
                display_string = ":" + codepoint_to_name[codepoint] + ":"
                to_process.append(
                    {
                        "type": "emoji",
                        "start": match.start(),
                        "end": match.end(),
                        "codepoint": codepoint,
                        "title": display_string,
                    }
                )
        to_process.sort(key=lambda x: x["start"])
        p = current_node = Element("p")

        def set_text(text: str) -> None:
            if current_node == p:
                current_node.text = text
            else:
                current_node.tail = text

        db_data: Optional[DbData] = self.zmd.zulip_db_data
        current_index = 0
        for item in to_process:
            if item["start"] < current_index:
                continue
            set_text(text[current_index : item["start"]])
            current_index = item["end"]
            if item["type"] != "emoji":
                elem = url_to_a(db_data, item["url"], item["text"])
                assert isinstance(elem, Element)
            else:
                elem = make_emoji(item["codepoint"], item["title"])
            current_node = elem
            p.append(elem)
        set_text(text[current_index:])
        return p

    def twitter_link(self, url: str) -> Optional[Element]:
        tweet_id = get_tweet_id(url)
        if tweet_id is None:
            return None
        try:
            res = fetch_tweet_data(tweet_id)
            if res is None:
                return None
            user: Dict[str, Any] = res["user"]
            tweet = Element("div")
            tweet.set("class", "twitter-tweet")
            img_a = SubElement(tweet, "a")
            img_a.set("href", url)
            profile_img = SubElement(img_a, "img")
            profile_img.set("class", "twitter-avatar")
            image_url = user.get("profile_image_url_https", user["profile_image_url"])
            profile_img.set("src", image_url)
            text = html.unescape(res["full_text"])
            urls = res.get("urls", [])
            user_mentions = res.get("user_mentions", [])
            media: List[Dict[str, Any]] = res.get("media", [])
            p = self.twitter_text(text, urls, user_mentions, media)
            tweet.append(p)
            span = SubElement(tweet, "span")
            span.text = "- {} (@{})".format(user["name"], user["screen_name"])
            for media_item in media:
                if media_item["type"] != "photo":
                    continue
                size_name_tuples = sorted(
                    media_item["sizes"].items(), reverse=True, key=lambda x: x[1]["h"]
                )
                for size_name, size in size_name_tuples:
                    if size["h"] < self.TWITTER_MAX_IMAGE_HEIGHT:
                        break
                media_url = "{}:{}".format(media_item["media_url_https"], size_name)
                img_div = SubElement(tweet, "div")
                img_div.set("class", "twitter-image")
                img_a = SubElement(img_div, "a")
                img_a.set("href", media_item["url"])
                img = SubElement(img_a, "img")
                img.set("src", media_url)
            return tweet
        except NotImplementedError:
            return None
        except Exception:
            markdown_logger.warning("Error building Twitter link", exc_info=True)
            return None

    def get_url_data(self, e: Element) -> Optional[Tuple[str, Optional[str]]]:
        if e.tag == "a":
            url = e.get("href")
            assert url is not None
            return (url, e.text)
        return None

    def get_inlining_information(
        self,
        root: Element,
        found_url: ResultWithFamily[Tuple[str, Optional[str]]],
    ) -> LinkInfo:
        grandparent = found_url.family.grandparent
        parent = found_url.family.parent
        ahref_element = found_url.family.child
        (url, text) = found_url.result
        url_eq_text = text is None or url == text
        title = None if url_eq_text else text
        info: LinkInfo = {
            "parent": root,
            "title": title,
            "index": None,
            "remove": None,
        }
        if parent.tag == "li":
            info["parent"] = parent
            if not parent.text and not ahref_element.tail and url_eq_text:
                info["remove"] = ahref_element
        elif parent.tag == "p":
            assert grandparent is not None
            parent_index = None
            for index, uncle in enumerate(grandparent):
                if uncle is parent:
                    parent_index = index
                    break
            info["parent"] = grandparent
            if (
                len(parent) == 1
                and (not parent.text or parent.text == "\n")
                and not ahref_element.tail
                and url_eq_text
            ):
                info["remove"] = parent
            if parent_index is not None:
                info["