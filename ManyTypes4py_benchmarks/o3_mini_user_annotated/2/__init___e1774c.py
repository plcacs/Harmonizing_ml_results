#!/usr/bin/env python3
"""
Zulip's main Markdown implementation.  See docs/subsystems/markdown.md for
detailed documentation on our Markdown syntax.
"""

import html
import logging
import mimetypes
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from functools import lru_cache
from re import Match, Pattern
from typing import Any, Generic, Optional, TypeAlias, TypedDict, TypeVar, cast

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
_T = TypeVar("_T")
ElementStringNone: TypeAlias = Element | str | None


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
    mentions_user_ids: set[int]
    mentions_user_group_ids: set[int]
    alert_words: set[str]
    links_for_preview: set[str]
    user_ids_with_alert_words: set[int]
    potential_attachment_path_ids: list[str]
    thumbnail_spinners: set[str]


@dataclass
class DbData:
    mention_data: MentionData
    realm_url: str
    realm_alert_words_automaton: Optional[ahocorasick.Automaton]
    active_realm_emoji: dict[str, EmojiInfo]
    sent_by_bot: bool
    stream_names: dict[str, int]
    topic_info: dict[ChannelTopicInfo, Optional[int]]
    translate_emoticons: bool
    user_upload_previews: dict[str, MarkdownImageMetadata]


version = 1


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
                    [^\s()\"]*?            
                    (?: \( %s \)           
                        [^\s()\"]*?        
                    )*                     
                   """
    nested_paren_chunk = paren_group
    for i in range(6):
        nested_paren_chunk %= (paren_group,)
    nested_paren_chunk %= (inner_paren_contents,)

    file_links = r"| (?:file://(/[^/ ]*)+/?)" if settings.ENABLE_FILE_LINKS else r""
    REGEX = rf"""
        (?<![^\s'"\(,:<])    
        (?P<url>             
            (?:(?:           
                https?://[\w.:@-]+?   
               |(?:                   
                    (?:[\w-]+\.)+     
                    (?:{tlds})        
                )
            )
            (?:/             
                {nested_paren_chunk}           
            )?)              
            | (?:[\w.-]+\@[\w.-]+\.[\w]+) 
            {file_links}               
            | (?:bitcoin:[13][a-km-zA-HJ-NP-Z1-9]{{25,34}})
        )
        (?=                            
            [!:;\?\),\.\'\"\>]*         
            (?:\Z|\s)                  
        )
        """
    return verbose_compile(REGEX)


def clear_web_link_regex_for_testing() -> None:
    get_web_link_regex.cache_clear()


markdown_logger: logging.Logger = logging.getLogger()


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


def list_of_tlds() -> list[str]:
    common_false_positives: set[str] = {"java", "md", "mov", "py", "zip"}
    return sorted(tld_set - common_false_positives, key=len, reverse=True)


def walk_tree(
    root: Element, processor: Callable[[Element], Optional[_T]], stop_after_first: bool = False
) -> list[_T]:
    results: list[_T] = []
    queue: deque[Element] = deque([root])
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
) -> list[ResultWithFamily[_T]]:
    results: list[ResultWithFamily[_T]] = []
    queue: deque[ElementPair] = deque([ElementPair(parent=None, value=root)])
    while queue:
        currElementPair = queue.popleft()
        for child in currElementPair.value:
            queue.append(ElementPair(parent=currElementPair, value=child))
            result = processor(child)
            if result is not None:
                if currElementPair.parent is not None:
                    grandparent: Optional[Element] = currElementPair.parent.value
                else:
                    grandparent = None
                family: ElementFamily = ElementFamily(
                    grandparent=grandparent,
                    parent=currElementPair.value,
                    child=child,
                    in_blockquote=has_blockquote_ancestor(currElementPair),
                )
                results.append(ResultWithFamily(family=family, result=result))
        # end for
    return results


def has_blockquote_ancestor(element_pair: Optional[ElementPair]) -> bool:
    if element_pair is None:
        return False
    elif element_pair.value.tag == "blockquote":
        return True
    else:
        return has_blockquote_ancestor(element_pair.parent)


@cache_with_key(lambda tweet_id: tweet_id, cache_name="database")
def fetch_tweet_data(tweet_id: str) -> Optional[dict[str, Any]]:
    raise NotImplementedError("Twitter desupported their v1 API")


class OpenGraphSession(OutgoingSession):
    def __init__(self) -> None:
        super().__init__(role="markdown", timeout=1)


def fetch_open_graph_image(url: str) -> Optional[dict[str, Any]]:
    og: dict[str, Optional[str]] = {"image": None, "title": None, "desc": None}
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
            html_flag: bool = mimetype == "text/html"
            res.raw.decode_content = True
            for event, element in lxml.etree.iterparse(
                res.raw, events=("start",), no_network=True, remove_comments=True, html=html_flag
            ):
                parent = element.getparent()
                if parent is not None:
                    parent.text = None
                    parent.remove(element)
                if element.tag in ("body", "{http://www.w3.org/1999/xhtml}body"):
                    break
                elif element.tag in ("meta", "{http://www.w3.org/1999/xhtml}meta"):
                    if element.get("property") == "og:image":
                        content: Optional[str] = element.get("content")
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
    to_match: str = parsed_url.path
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
        found_imgs: list[Element] = walk_tree(root, lambda e: e if e.tag == "img" else None)
        for img in found_imgs:
            url: Optional[str] = img.get("src")
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
        found_videos: list[Element] = walk_tree(root, lambda e: e if e.tag == "video" else None)
        for video in found_videos:
            url: Optional[str] = video.get("src")
            assert url is not None
            if is_static_or_current_realm_url(url, self.zmd.zulip_realm):
                continue
            video.set("src", get_camo_url(url))


class BacktickInlineProcessor(markdown.inlinepatterns.BacktickInlineProcessor):
    @override
    def handleMatch(self, m: Match[str], data: str) -> tuple[Element | str | None, int | None, int | None]:
        el, start, end = ret = super().handleMatch(m, data)
        if el is not None and m.group(3):
            assert isinstance(el, Element)
            el.text = markdown.util.AtomicString(markdown.util.code_escape(m.group(3)))
        return ret


IMAGE_EXTENSIONS: list[str] = [".bmp", ".gif", ".jpe", ".jpeg", ".jpg", ".png", ".webp"]


class InlineInterestingLinkProcessor(markdown.treeprocessors.Treeprocessor):
    TWITTER_MAX_IMAGE_HEIGHT: int = 400
    TWITTER_MAX_TO_PREVIEW: int = 3
    INLINE_PREVIEW_LIMIT_PER_MESSAGE: int = 24

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
            path_id: str = image_url.removeprefix("/user_uploads/")
            assert path_id in self.zmd.zulip_db_data.user_upload_previews
            metadata: MarkdownImageMetadata = self.zmd.zulip_db_data.user_upload_previews[path_id]
            img.set("class", "image-loading-placeholder")
            img.set("src", "/static/images/loading/loader-black.svg")
            img.set("data-original-dimensions", f"{metadata.original_width_px}x{metadata.original_height_px}")
            if metadata.original_content_type:
                img.set("data-original-content-type", metadata.original_content_type)
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
        img_link: str = get_camo_url(extracted_data.image)
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
                return urljoin("https://raw.githubusercontent.com", "/".join(split_path[0:3] + split_path[4:]))
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
        if parsed_url.netloc.lower().endswith(".wikipedia.org") and parsed_url.path.startswith("/wiki/File:"):
            newpath = parsed_url.path.replace("/wiki/File:", "/wiki/Special:FilePath/File:", 1)
            return parsed_url._replace(path=newpath).geturl()
        if parsed_url.netloc == "linx.li":
            return "https://linx.li/s" + parsed_url.path
        return None

    def dropbox_image(self, url: str) -> Optional[dict[str, Any]]:
        parsed_url = urlsplit(url)
        if parsed_url.netloc == "dropbox.com" or parsed_url.netloc.endswith(".dropbox.com"):
            is_album = parsed_url.path.startswith("/sc/") or parsed_url.path.startswith("/photos/")
            if not (parsed_url.path.startswith("/s/") or parsed_url.path.startswith("/sh/") or is_album):
                return None
            image_info: Optional[dict[str, Any]] = fetch_open_graph_image(url)
            is_image: bool = is_album or self.is_image(url)
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
        yt_id: Optional[str] = None
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
                    yt_id = query["v"][0]
                elif split_url.path == "/watch_videos" and "video_ids" in query:
                    yt_id = query["video_ids"][0].split(",", 1)[0]
                elif split_url.path.startswith(("/embed/", "/shorts/", "/v/")):
                    yt_id = split_url.path.split("/", 3)[2]
            elif split_url.hostname == "youtu.be" and split_url.path.startswith("/"):
                yt_id = split_url.path.removeprefix("/")
        if yt_id is not None and re.fullmatch(r"[0-9A-Za-z_-]+", yt_id):
            return yt_id
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
            r"(?:channels\/(?:\w+\/)?|groups\/([^\/]*)\/videos\/|)(\d+)(?:|\/\?))$"
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
        urls: list[dict[str, str]],
        user_mentions: list[dict[str, Any]],
        media: list[dict[str, Any]],
    ) -> Element:
        to_process: list[dict[str, Any]] = []
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
            screen_name: str = user_mention["screen_name"]
            mention_string: str = "@" + screen_name
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
            short_url: str = media_item["url"]
            expanded_url: str = media_item["expanded_url"]
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
            orig_syntax: str = match.group("syntax")
            codepoint: str = emoji_to_hex_codepoint(unqualify_emoji(orig_syntax))
            if codepoint in codepoint_to_name:
                display_string: str = ":" + codepoint_to_name[codepoint] + ":"
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
        p: Element = Element("p")
        current_node: Element = p
        def set_text(text_to_set: str) -> None:
            if current_node == p:
                current_node.text = text_to_set
            else:
                current_node.tail = text_to_set
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        current_index: int = 0
        for item in to_process:
            if item["start"] < current_index:
                continue
            set_text(text[current_index:item["start"]])
            current_index = item["end"]
            if item["type"] != "emoji":
                elem: Element = url_to_a(db_data, item["url"], item["text"])  # type: ignore
                assert isinstance(elem, Element)
            else:
                elem = make_emoji(item["codepoint"], item["title"])
            current_node = elem
            p.append(elem)
        set_text(text[current_index:])
        return p

    def twitter_link(self, url: str) -> Optional[Element]:
        tweet_id: Optional[str] = get_tweet_id(url)
        if tweet_id is None:
            return None
        try:
            res: Optional[dict[str, Any]] = fetch_tweet_data(tweet_id)
            if res is None:
                return None
            user: dict[str, Any] = res["user"]
            tweet: Element = Element("div")
            tweet.set("class", "twitter-tweet")
            img_a: Element = SubElement(tweet, "a")
            img_a.set("href", url)
            profile_img: Element = SubElement(img_a, "img")
            profile_img.set("class", "twitter-avatar")
            image_url: str = user.get("profile_image_url_https", user["profile_image_url"])
            profile_img.set("src", image_url)
            text_content: str = html.unescape(res["full_text"])
            urls_list: list[dict[str, str]] = res.get("urls", [])
            user_mentions: list[dict[str, Any]] = res.get("user_mentions", [])
            media: list[dict[str, Any]] = res.get("media", [])
            p: Element = self.twitter_text(text_content, urls_list, user_mentions, media)
            tweet.append(p)
            span: Element = SubElement(tweet, "span")
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
                media_url: str = f"{media_item['media_url_https']}:{size_name}"
                img_div: Element = SubElement(tweet, "div")
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

    def get_url_data(self, e: Element) -> Optional[tuple[str, Optional[str]]]:
        if e.tag == "a":
            url: Optional[str] = e.get("href")
            assert url is not None
            return (url, e.text)
        return None

    def get_inlining_information(
        self,
        root: Element,
        found_url: ResultWithFamily[tuple[str, Optional[str]]],
    ) -> LinkInfo:
        grandparent: Optional[Element] = found_url.family.grandparent
        parent: Element = found_url.family.parent
        ahref_element: Element = found_url.family.child
        url, text = found_url.result
        url_eq_text: bool = text is None or url == text
        title: Optional[str] = None if url_eq_text else text
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
            parent_index: Optional[int] = None
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
                info["index"] = self.find_proper_insertion_index(grandparent, parent, parent_index)
        return info

    def handle_image_inlining(
        self,
        root: Element,
        found_url: ResultWithFamily[tuple[str, Optional[str]]],
    ) -> None:
        info: LinkInfo = self.get_inlining_information(root, found_url)
        url, text = found_url.result
        actual_url: str = self.get_actual_image_url(url)
        self.add_a(
            info["parent"],
            image_url=actual_url,
            link=url,
            title=info["title"],
            insertion_index=info["index"],
        )
        if info["remove"] is not None:
            info["parent"].remove(info["remove"])

    def handle_tweet_inlining(
        self,
        root: Element,
        found_url: ResultWithFamily[tuple[str, Optional[str]]],
        twitter_data: Element,
    ) -> None:
        info: LinkInfo = self.get_inlining_information(root, found_url)
        if info["index"] is not None:
            div: Element = Element("div")
            root.insert(info["index"], div)
        else:
            div = SubElement(root, "div")
        div.set("class", "inline-preview-twitter")
        div.insert(0, twitter_data)

    def handle_youtube_url_inlining(
        self,
        root: Element,
        found_url: ResultWithFamily[tuple[str, Optional[str]]],
        yt_image: str,
    ) -> None:
        info: LinkInfo = self.get_inlining_information(root, found_url)
        url, text = found_url.result
        yt_id: Optional[str] = self.youtube_id(url)
        self.add_a(
            info["parent"],
            image_url=yt_image,
            link=url,
            class_attr="youtube-video message_inline_image",
            data_id=yt_id,
            insertion_index=info["index"],
            already_thumbnailed=True,
        )

    def find_proper_insertion_index(
        self, grandparent: Element, parent: Element, parent_index_in_grandparent: int
    ) -> int:
        parent_links = [ele.attrib["href"] for ele in parent.iter(tag="a")]
        insertion_index: int = parent_index_in_grandparent
        while True:
            insertion_index += 1
            if insertion_index >= len(grandparent):
                return insertion_index
            uncle: Element = grandparent[insertion_index]
            inline_image_classes: set[str] = {"message_inline_image", "message_inline_ref", "inline-preview-twitter"}
            if (
                uncle.tag != "div"
                or "class" not in uncle.attrib
                or not (set(uncle.attrib["class"].split()) & inline_image_classes)
            ):
                return insertion_index
            uncle_link: Optional[Element] = uncle.find("a")
            assert uncle_link is not None
            if uncle_link.attrib["href"] not in parent_links:
                return insertion_index

    def is_video(self, url: str) -> bool:
        if not self.zmd.image_preview_enabled:
            return False
        url_type: Optional[str] = mimetypes.guess_type(url)[0]
        supported_mimetypes: list[str] = ["video/mp4", "video/webm"]
        return url_type in supported_mimetypes

    def add_video(
        self,
        root: Element,
        url: str,
        title: Optional[str],
        class_attr: str = "message_inline_image message_inline_video",
        insertion_index: Optional[int] = None,
    ) -> None:
        if insertion_index is not None:
            div: Element = Element("div")
            root.insert(insertion_index, div)
        else:
            div = SubElement(root, "div")
        div.set("class", class_attr)
        a: Element = SubElement(div, "a")
        a.set("href", url)
        if title:
            a.set("title", title)
        video: Element = SubElement(a, "video")
        video.set("src", url)
        video.set("preload", "metadata")

    def handle_video_inlining(
        self, root: Element, found_url: ResultWithFamily[tuple[str, Optional[str]]]
    ) -> None:
        info: LinkInfo = self.get_inlining_information(root, found_url)
        url: str = found_url.result[0]
        self.add_video(info["parent"], url, info["title"], insertion_index=info["index"])
        if info["remove"] is not None:
            info["parent"].remove(info["remove"])

    @override
    def run(self, root: Element) -> None:
        found_urls: list[ResultWithFamily[tuple[str, Optional[str]]]] = walk_tree_with_family(root, self.get_url_data)
        unique_urls: set[str] = {found_url.result[0] for found_url in found_urls}
        unique_previewable_urls: set[str] = {found_url.result[0] for found_url in found_urls if not found_url.family.in_blockquote}
        if self.zmd.zulip_message:
            self.zmd.zulip_message.has_link = len(found_urls) > 0
            self.zmd.zulip_message.has_image = False
            for url in unique_urls:
                parsed_url = urlsplit(urljoin("/", url))
                host = parsed_url.netloc
                if host != "" and (self.zmd.zulip_realm is None or host != self.zmd.zulip_realm.host):
                    continue
                if not parsed_url.path.startswith("/user_uploads/"):
                    continue
                path_id = parsed_url.path.removeprefix("/user_uploads/")
                self.zmd.zulip_rendering_result.potential_attachment_path_ids.append(path_id)
        if len(found_urls) == 0:
            return
        if len(unique_previewable_urls) > self.INLINE_PREVIEW_LIMIT_PER_MESSAGE:
            return
        processed_urls: set[str] = set()
        rendered_tweet_count: int = 0
        for found_url in found_urls:
            url, text = found_url.result
            if url in unique_previewable_urls and url not in processed_urls:
                processed_urls.add(url)
            else:
                continue
            if self.is_video(url):
                self.handle_video_inlining(root, found_url)
                continue
            dropbox_image = self.dropbox_image(url)
            if dropbox_image is not None:
                class_attr: str = "message_inline_ref"
                is_image: bool = dropbox_image["is_image"]
                if is_image:
                    class_attr = "message_inline_image"
                self.add_a(
                    root,
                    image_url=dropbox_image["image"],
                    link=url,
                    title=dropbox_image.get("title"),
                    desc=dropbox_image.get("desc", ""),
                    class_attr=class_attr,
                    already_thumbnailed=True,
                )
                continue
            if self.is_image(url):
                image_source: Optional[str] = self.corrected_image_source(url)
                if image_source is not None:
                    found_url = ResultWithFamily(
                        family=found_url.family,
                        result=(image_source, image_source),
                    )
                self.handle_image_inlining(root, found_url)
                continue
            netloc: str = urlsplit(url).netloc
            if netloc == "" or (self.zmd.zulip_realm is not None and netloc == self.zmd.zulip_realm.host):
                continue
            if get_tweet_id(url) is not None:
                if rendered_tweet_count >= self.TWITTER_MAX_TO_PREVIEW:
                    continue
                twitter_data: Optional[Element] = self.twitter_link(url)
                if twitter_data is None:
                    continue
                rendered_tweet_count += 1
                self.handle_tweet_inlining(root, found_url, twitter_data)
                continue
            youtube = self.youtube_image(url)
            if youtube is not None:
                self.handle_youtube_url_inlining(root, found_url, youtube)
            db_data: Optional[DbData] = self.zmd.zulip_db_data
            if db_data and db_data.sent_by_bot:
                continue
            if not self.zmd.url_embed_preview_enabled:
                continue
            if self.zmd.url_embed_data is None or url not in self.zmd.url_embed_data:
                self.zmd.zulip_rendering_result.links_for_preview.add(url)
                continue
            extracted_data = self.zmd.url_embed_data[url]
            if extracted_data is None:
                continue
            if youtube is not None:
                title = self.youtube_title(extracted_data)
                if title is not None:
                    if url == text:
                        found_url.family.child.text = title
                    else:
                        found_url.family.child.text = text
                continue
            self.add_embed(root, url, extracted_data)
            if self.vimeo_id(url):
                title = self.vimeo_title(extracted_data)
                if title:
                    if url == text:
                        found_url.family.child.text = title
                    else:
                        found_url.family.child.text = text


class CompiledInlineProcessor(markdown.inlinepatterns.InlineProcessor):
    def __init__(self, compiled_re: Pattern[str], zmd: "ZulipMarkdown") -> None:
        self.compiled_re: Pattern[str] = compiled_re
        self.md = zmd
        self.zmd = zmd


class Timestamp(markdown.inlinepatterns.Pattern):
    @override
    def handleMatch(self, match: Match[str]) -> Optional[Element]:
        time_input_string: str = match.group("time")
        try:
            timestamp: datetime = dateutil.parser.parse(time_input_string, tzinfos=common_timezones)
        except (ValueError, OverflowError):
            try:
                timestamp = datetime.fromtimestamp(float(time_input_string), tz=timezone.utc)
            except ValueError:
                timestamp = None
        if not timestamp:
            error_element: Element = Element("span")
            error_element.set("class", "timestamp-error")
            error_element.text = markdown.util.AtomicString(f"Invalid time format: {time_input_string}")
            return error_element
        time_element: Element = Element("time")
        if timestamp.tzinfo:
            try:
                timestamp = timestamp.astimezone(timezone.utc)
            except (ValueError, OverflowError):
                error_element = Element("span")
                error_element.set("class", "timestamp-error")
                error_element.text = markdown.util.AtomicString(f"Invalid time format: {time_input_string}")
                return error_element
        else:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        time_element.set("datetime", timestamp.isoformat().replace("+00:00", "Z"))
        time_element.text = markdown.util.AtomicString(time_input_string)
        return time_element


POSSIBLE_EMOJI_RE: Pattern[str] = regex.compile(
    r"""(?P<syntax>
\p{RI} \p{RI}
| \p{Emoji}
  (?: \p{Emoji_Modifier}
  | \uFE0F \u20E3?
  | [\U000E0020-\U000E007E]+ \U000E007F
  )?
  (?: \u200D
    (?: \p{RI} \p{RI}
    | \p{Emoji}
      (?: \p{Emoji_Modifier}
      | \uFE0F \u20E3?
      | [\U000E0020-\U000E007E]+ \U000E007F
      )?
    )
  )*)
""",
    regex.VERBOSE,
)


def make_emoji(codepoint: str, display_string: str) -> Element:
    title: str = display_string[1:-1].replace("_", " ")
    span: Element = Element("span")
    span.set("class", f"emoji emoji-{codepoint}")
    span.set("title", title)
    span.set("role", "img")
    span.set("aria-label", title)
    span.text = markdown.util.AtomicString(display_string)
    return span


def make_realm_emoji(src: str, display_string: str) -> Element:
    elt: Element = Element("img")
    elt.set("src", src)
    elt.set("class", "emoji")
    elt.set("alt", display_string)
    elt.set("title", display_string[1:-1].replace("_", " "))
    return elt


class EmoticonTranslation(markdown.inlinepatterns.Pattern):
    def __init__(self, pattern: str, zmd: "ZulipMarkdown") -> None:
        super().__init__(pattern, zmd)
        self.zmd = zmd

    @override
    def handleMatch(self, match: Match[str]) -> Optional[Element]:
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        if db_data is None or not db_data.translate_emoticons:
            return None
        emoticon: str = match.group("emoticon")
        translated: str = translate_emoticons(emoticon)
        name: str = translated[1:-1]
        return make_emoji(name_to_codepoint[name], translated)


TEXT_PRESENTATION_RE: Pattern[str] = regex.compile(r"\P{Emoji_Presentation}\u20E3?")


class UnicodeEmoji(CompiledInlineProcessor):
    @override
    def handleMatch(self, match: Match[str], data: str) -> tuple[Optional[Element], Optional[int], Optional[int]]:
        orig_syntax: str = match.group("syntax")
        if TEXT_PRESENTATION_RE.fullmatch(orig_syntax):
            return None, None, None
        codepoint: str = emoji_to_hex_codepoint(unqualify_emoji(orig_syntax))
        if codepoint in codepoint_to_name:
            display_string: str = ":" + codepoint_to_name[codepoint] + ":"
            return make_emoji(codepoint, display_string), match.start(), match.end()
        else:
            return None, None, None


class Emoji(markdown.inlinepatterns.Pattern):
    def __init__(self, pattern: str, zmd: "ZulipMarkdown") -> None:
        super().__init__(pattern, zmd)
        self.zmd = zmd

    @override
    def handleMatch(self, match: Match[str]) -> Optional[str | Element]:
        orig_syntax: str = match.group("syntax")
        name: str = orig_syntax[1:-1]
        active_realm_emoji: dict[str, EmojiInfo] = {}
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        if db_data is not None:
            active_realm_emoji = db_data.active_realm_emoji
        if name in active_realm_emoji:
            return make_realm_emoji(active_realm_emoji[name]["source_url"], orig_syntax)
        elif name == "zulip":
            return make_realm_emoji("/static/generated/emoji/images/emoji/unicode/zulip.png", orig_syntax)
        elif name in name_to_codepoint:
            return make_emoji(name_to_codepoint[name], orig_syntax)
        else:
            return orig_syntax


def content_has_emoji_syntax(content: str) -> bool:
    return re.search(EMOJI_REGEX, content) is not None


class Tex(markdown.inlinepatterns.Pattern):
    @override
    def handleMatch(self, match: Match[str]) -> str | Element:
        rendered: Optional[str] = render_tex(match.group("body"), is_inline=True)
        if rendered is not None:
            return self.md.htmlStash.store(rendered)
        else:
            span: Element = Element("span")
            span.set("class", "tex-error")
            span.text = markdown.util.AtomicString("$$" + match.group("body") + "$$")
            return span


def sanitize_url(url: str) -> Optional[str]:
    try:
        parts = urlsplit(url.replace(" ", "%20"))
        scheme, netloc, path, query, fragment = parts
    except ValueError:
        return ""
    if scheme == "" and netloc == "" and "@" in path:
        scheme = "mailto"
    elif scheme == "" and netloc == "" and len(path) > 0 and path[0] == "/":
        return urlunsplit(("", "", path, query, fragment))
    elif (scheme, netloc, path, query) == ("", "", "", "") and len(fragment) > 0:
        return urlunsplit(("", "", "", "", fragment))
    if not scheme:
        return sanitize_url("http://" + url)
    if scheme not in allowed_schemes:
        return None
    return urlunsplit((scheme, netloc, path, query, fragment))


def url_to_a(db_data: Optional[DbData], url: str, text: Optional[str] = None) -> Element | str:
    a: Element = Element("a")
    href: Optional[str] = sanitize_url(url)
    if href is None:
        return url
    if text is None:
        text = markdown.util.AtomicString(url)
    href = rewrite_local_links_to_relative(db_data, href)
    a.set("href", href)
    a.text = text
    return a


class CompiledPattern(markdown.inlinepatterns.Pattern):
    def __init__(self, compiled_re: Pattern[str], zmd: "ZulipMarkdown") -> None:
        self.compiled_re: Pattern[str] = compiled_re
        self.md = zmd
        self.zmd = zmd


class AutoLink(CompiledPattern):
    @override
    def handleMatch(self, match: Match[str]) -> ElementStringNone:
        url: str = match.group("url")
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        return url_to_a(db_data, url)


class OListProcessor(sane_lists.SaneOListProcessor):
    def __init__(self, parser: BlockParser) -> None:
        parser.md.tab_length = 2
        super().__init__(parser)
        parser.md.tab_length = 4


class UListProcessor(sane_lists.SaneUListProcessor):
    def __init__(self, parser: BlockParser) -> None:
        parser.md.tab_length = 2
        super().__init__(parser)
        parser.md.tab_length = 4


class ListIndentProcessor(markdown.blockprocessors.ListIndentProcessor):
    def __init__(self, parser: BlockParser) -> None:
        parser.md.tab_length = 2
        super().__init__(parser)
        parser.md.tab_length = 4


class HashHeaderProcessor(markdown.blockprocessors.HashHeaderProcessor):
    RE = re.compile(r"(?:^|\n)(?P<level>#{1,6})\s(?P<header>(?:\\.|[^\\])*?)#*(?:\n|$)")


class BlockQuoteProcessor(markdown.blockprocessors.BlockQuoteProcessor):
    RE = re.compile(r"(^|\n)(?!(?:[ ]{0,3}>\s*(?:$|\n))*(?:$|\n))[ ]{0,3}>[ ]?(.*)")

    @override
    def run(self, parent: Element, blocks: list[str]) -> None:
        block: str = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            before: str = block[: m.start()]
            self.parser.parseBlocks(parent, [before])
            block = "\n".join([self.clean(line) for line in block[m.start() :].split("\n")])
        quote: Element = SubElement(parent, "blockquote")
        self.parser.state.set("blockquote")
        self.parser.parseChunk(quote, block)
        self.parser.state.reset()

    @override
    def clean(self, line: str) -> str:
        line = mention.MENTIONS_RE.sub(lambda m: "@_**{}**".format(m.group("match")), line)
        line = mention.USER_GROUP_MENTIONS_RE.sub(lambda m: "@_*{}*".format(m.group("match")), line)
        return super().clean(line)


@dataclass
class Fence:
    fence_str: str
    is_code: bool


class MarkdownListPreprocessor(markdown.preprocessors.Preprocessor):
    LI_RE: Pattern[str] = re.compile(r"^[ ]*([*+-]|\d\.)[ ]+(.*)", re.MULTILINE)

    @override
    def run(self, lines: list[str]) -> list[str]:
        inserts: int = 0
        in_code_fence: bool = False
        open_fences: list[Fence] = []
        copy: list[str] = lines[:]
        for i in range(len(lines) - 1):
            m = FENCE_RE.match(lines[i])
            if m:
                fence_str: str = m.group("fence")
                lang: Optional[str] = m.group("lang")
                is_code: bool = lang not in ("quote", "quoted")
                matches_last_fence: bool = (fence_str == open_fences[-1].fence_str) if open_fences else False
                closes_last_fence: bool = not lang and matches_last_fence
                if closes_last_fence:
                    open_fences.pop()
                else:
                    open_fences.append(Fence(fence_str, is_code))
                in_code_fence = any(fence.is_code for fence in open_fences)
            li1 = self.LI_RE.match(lines[i])
            li2 = self.LI_RE.match(lines[i + 1])
            if (
                not in_code_fence
                and lines[i]
                and ((li2 and not li1) or (li1 and li2 and (len(li1.group(1)) == 1) != (len(li2.group(1)) == 1)))
            ):
                copy.insert(i + inserts + 1, "")
                inserts += 1
        return copy


BEFORE_CAPTURE_GROUP: str = "linkifier_before_match"
OUTER_CAPTURE_GROUP: str = "linkifier_actual_match"
AFTER_CAPTURE_GROUP: str = "linkifier_after_match"


def prepare_linkifier_pattern(source: str) -> str:
    next_line: str = "\u0085"
    return rf"""(?P<{BEFORE_CAPTURE_GROUP}>^|\s|{next_line}|\pZ|['"\(,:<])(?P<{OUTER_CAPTURE_GROUP}>{source})(?P<{AFTER_CAPTURE_GROUP}>$|[^\pL\pN])"""


class LinkifierPattern(CompiledInlineProcessor):
    def __init__(self, source_pattern: str, url_template: str, zmd: "ZulipMarkdown") -> None:
        options = re2.Options()
        options.log_errors = False
        compiled_re2: Pattern[str] = re2.compile(prepare_linkifier_pattern(source_pattern), options=options)
        self.prepared_url_template = uri_template.URITemplate(url_template)
        super().__init__(compiled_re2, zmd)

    @override
    def handleMatch(self, m: Match[str], data: str) -> tuple[Element | str | None, int | None, int | None]:
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        url_elem: Element | str = url_to_a(db_data, self.prepared_url_template.expand(**m.groupdict()), markdown.util.AtomicString(m.group(OUTER_CAPTURE_GROUP)))
        if isinstance(url_elem, str):
            return None, None, None
        return url_elem, m.start(2), m.end(2)


class UserMentionPattern(CompiledInlineProcessor):
    @override
    def handleMatch(self, m: Match[str], data: str) -> tuple[Element | str | None, int | None, int | None]:
        name: str = m.group("match")
        silent: bool = m.group("silent") == "_"
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        if db_data is not None:
            topic_wildcard: bool = mention.user_mention_matches_topic_wildcard(name)
            stream_wildcard: bool = mention.user_mention_matches_stream_wildcard(name)
            id_syntax_match = re.match(r"(?P<full_name>.+)?\|(?P<user_id>\d+)$", name)
            if id_syntax_match:
                full_name: Optional[str] = id_syntax_match.group("full_name")
                id_val: int = int(id_syntax_match.group("user_id"))
                user = db_data.mention_data.get_user_by_id(id_val)
                if full_name and user and user.full_name != full_name:
                    return None, None, None
            else:
                user = db_data.mention_data.get_user_by_name(name)
            user_id: Optional[str] = None
            if stream_wildcard:
                if not silent:
                    self.zmd.zulip_rendering_result.mentions_stream_wildcard = True
                user_id = "*"
            elif topic_wildcard:
                if not silent:
                    self.zmd.zulip_rendering_result.mentions_topic_wildcard = True
            elif user is not None:
                assert isinstance(user, FullNameInfo)
                if not user.is_active:
                    silent = True
                if not silent:
                    self.zmd.zulip_rendering_result.mentions_user_ids.add(user.id)
                name = user.full_name
                user_id = str(user.id)
            else:
                return None, None, None
            el: Element = Element("span")
            if user_id:
                el.set("data-user-id", user_id)
            text_val: str = f"@{name}"
            if topic_wildcard:
                el.set("class", "topic-mention")
            elif stream_wildcard:
                el.set("class", "user-mention channel-wildcard-mention")
            else:
                el.set("class", "user-mention")
            if silent:
                el.set("class", el.get("class", "") + " silent")
                text_val = f"{name}"
            el.text = markdown.util.AtomicString(text_val)
            return el, m.start(), m.end()
        return None, None, None


class UserGroupMentionPattern(CompiledInlineProcessor):
    @override
    def handleMatch(self, m: Match[str], data: str) -> tuple[Element | str | None, int | None, int | None]:
        name: str = m.group("match")
        silent: bool = m.group("silent") == "_"
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        if db_data is not None:
            user_group = db_data.mention_data.get_user_group(name)
            if user_group:
                if user_group.deactivated:
                    silent = True
                if not silent:
                    self.zmd.zulip_rendering_result.mentions_user_group_ids.add(user_group.id)
                name = get_user_group_mention_display_name(user_group)
                user_group_id: str = str(user_group.id)
            else:
                return None, None, None
            el: Element = Element("span")
            el.set("data-user-group-id", user_group_id)
            if silent:
                el.set("class", "user-group-mention silent")
                text_val = f"{name}"
            else:
                el.set("class", "user-group-mention")
                text_val = f"@{name}"
            el.text = markdown.util.AtomicString(text_val)
            return el, m.start(), m.end()
        return None, None, None


class StreamTopicMessageProcessor(CompiledInlineProcessor):
    def find_stream_id(self, name: str) -> Optional[int]:
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        if db_data is None:
            return None
        stream_id: Optional[int] = db_data.stream_names.get(name)
        return stream_id


class StreamPattern(StreamTopicMessageProcessor):
    @override
    def handleMatch(self, m: Match[str], data: str) -> tuple[Element | str | None, int | None, int | None]:
        name: str = m.group("stream_name")
        stream_id: Optional[int] = self.find_stream_id(name)
        if stream_id is None:
            return None, None, None
        el: Element = Element("a")
        el.set("class", "stream")
        el.set("data-stream-id", str(stream_id))
        stream_url: str = encode_stream(stream_id, name)
        el.set("href", f"/#narrow/channel/{stream_url}")
        text_val: str = f"#{name}"
        el.text = markdown.util.AtomicString(text_val)
        return el, m.start(), m.end()


class StreamTopicPattern(StreamTopicMessageProcessor):
    def get_with_operand(self, channel_topic: ChannelTopicInfo) -> Optional[int]:
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        if db_data is None:
            return None
        with_operand: Optional[int] = db_data.topic_info.get(channel_topic)
        return with_operand

    @override
    def handleMatch(self, m: Match[str], data: str) -> tuple[Element | str | None, int | None, int | None]:
        stream_name: str = m.group("stream_name")
        topic_name: str = m.group("topic_name")
        stream_id: Optional[int] = self.find_stream_id(stream_name)
        if stream_id is None or topic_name is None:
            return None, None, None
        el: Element = Element("a")
        el.set("class", "stream-topic")
        el.set("data-stream-id", str(stream_id))
        stream_url: str = encode_stream(stream_id, stream_name)
        topic_url: str = hash_util_encode(topic_name)
        channel_topic_object: ChannelTopicInfo = ChannelTopicInfo(stream_name, topic_name)
        with_operand: Optional[int] = self.get_with_operand(channel_topic_object)
        if with_operand is not None:
            link: str = f"/#narrow/channel/{stream_url}/topic/{topic_url}/with/{with_operand}"
        else:
            link = f"/#narrow/channel/{stream_url}/topic/{topic_url}"
        el.set("href", link)
        if topic_name == "":
            topic_el: Element = Element("em")
            topic_el.text = Message.EMPTY_TOPIC_FALLBACK_NAME
            el.text = markdown.util.AtomicString(f"#{stream_name} > ")
            el.append(topic_el)
        else:
            text_val: str = f"#{stream_name} > {topic_name}"
            el.text = markdown.util.AtomicString(text_val)
        return el, m.start(), m.end()


class StreamTopicMessagePattern(StreamTopicMessageProcessor):
    @override
    def handleMatch(self, m: Match[str], data: str) -> tuple[Element | str | None, int | None, int | None]:
        stream_name: str = m.group("stream_name")
        topic_name: str = m.group("topic_name")
        message_id: str = m.group("message_id")
        stream_id: Optional[int] = self.find_stream_id(stream_name)
        if stream_id is None or topic_name is None:
            return None, None, None
        el: Element = Element("a")
        el.set("class", "message-link")
        stream_url: str = encode_stream(stream_id, stream_name)
        topic_url: str = hash_util_encode(topic_name)
        link: str = f"/#narrow/channel/{stream_url}/topic/{topic_url}/near/{message_id}"
        el.set("href", link)
        if topic_name == "":
            topic_el: Element = Element("em")
            topic_el.text = Message.EMPTY_TOPIC_FALLBACK_NAME
            el.text = markdown.util.AtomicString(f"#{stream_name} > ")
            el.append(topic_el)
            topic_el.tail = markdown.util.AtomicString(" @ 💬")
        else:
            text_val: str = f"#{stream_name} > {topic_name} @ 💬"
            el.text = markdown.util.AtomicString(text_val)
        return el, m.start(), m.end()


def possible_linked_stream_names(content: str) -> set[str]:
    return {
        *re.findall(STREAM_LINK_REGEX, content, re.VERBOSE),
        *(
            match.group("stream_name")
            for match in re.finditer(STREAM_TOPIC_LINK_REGEX, content, re.VERBOSE)
        ),
    }


def possible_linked_topics(content: str) -> set[ChannelTopicInfo]:
    return {
        ChannelTopicInfo(match.group("stream_name"), match.group("topic_name"))
        for match in re.finditer(STREAM_TOPIC_LINK_REGEX, content, re.VERBOSE)
    }


class AlertWordNotificationProcessor(markdown.preprocessors.Preprocessor):
    allowed_before_punctuation: set[str] = {" ", "\n", "(", '"', ".", ",", "'", ";", "[", "*", "`", ">"}
    allowed_after_punctuation: set[str] = {" ", "\n", ")", '",', "?", ":", ".", ",", "'", ";", "]", "!", "*", "`"}

    def __init__(self, zmd: "ZulipMarkdown") -> None:
        super().__init__(zmd)
        self.zmd = zmd

    def check_valid_start_position(self, content: str, index: int) -> bool:
        if index <= 0 or content[index] in self.allowed_before_punctuation:
            return True
        return False

    def check_valid_end_position(self, content: str, index: int) -> bool:
        if index >= len(content) or content[index] in self.allowed_after_punctuation:
            return True
        return False

    @override
    def run(self, lines: list[str]) -> list[str]:
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        if db_data is not None:
            realm_alert_words_automaton: Optional[ahocorasick.Automaton] = db_data.realm_alert_words_automaton
            if realm_alert_words_automaton is not None:
                content: str = "\n".join(lines).lower()
                for end_index, (original_value, user_ids) in realm_alert_words_automaton.iter(content):
                    if self.check_valid_start_position(content, end_index - len(original_value)) and self.check_valid_end_position(content, end_index + 1):
                        self.zmd.zulip_rendering_result.user_ids_with_alert_words.update(user_ids)
        return lines


class LinkInlineProcessor(markdown.inlinepatterns.LinkInlineProcessor):
    def __init__(self, pattern: str, zmd: "ZulipMarkdown") -> None:
        super().__init__(pattern, zmd)
        self.zmd = zmd

    def zulip_specific_link_changes(self, el: Element) -> Optional[Element]:
        href: Optional[str] = el.get("href")
        assert href is not None
        href = sanitize_url(self.unescape(href.strip()))
        if href is None:
            return None
        db_data: Optional[DbData] = self.zmd.zulip_db_data
        href = rewrite_local_links_to_relative(db_data, href)
        el.set("href", href)
        if not el.text or not el.text.strip():
            el.text = href
        el.text = markdown.util.AtomicString(el.text)
        return el

    @override
    def handleMatch(self, m: Match[str], data: str) -> tuple[Element | str | None, int | None, int | None]:
        ret = super().handleMatch(m, data)
        if ret[0] is not None:
            el, match_start, index = ret
            assert isinstance(el, Element)
            el = self.zulip_specific_link_changes(el)
            if el is not None:
                return el, match_start, index
        return None, None, None


def get_sub_registry(r: markdown.util.Registry[T], keys: list[str]) -> markdown.util.Registry[T]:
    new_r: markdown.util.Registry[T] = markdown.util.Registry[T]()
    for k in keys:
        new_r.register(r[k], k, r.get_index_for_name(k))
    return new_r


DEFAULT_MARKDOWN_KEY: int = -1
ZEPHYR_MIRROR_MARKDOWN_KEY: int = -2


class ZulipMarkdown(markdown.Markdown):
    zulip_message: Optional[Message]
    zulip_realm: Optional[Realm]
    zulip_db_data: Optional[DbData]
    zulip_rendering_result: MessageRenderingResult
    image_preview_enabled: bool
    url_embed_preview_enabled: bool
    url_embed_data: Optional[dict[str, Optional[UrlEmbedData]]]

    def __init__(self, linkifiers: list[LinkifierDict], linkifiers_key: int, email_gateway: bool) -> None:
        self.linkifiers = linkifiers
        self.linkifiers_key = linkifiers_key
        self.email_gateway = email_gateway
        super().__init__(extensions=[
            nl2br.makeExtension(),
            tables.makeExtension(),
            codehilite.makeExtension(
                linenums=False,
                guess_lang=False,
            ),
        ])
        self.set_output_format("html")

    @override
    def build_parser(self) -> Self:
        self.preprocessors = self.build_preprocessors()
        self.parser = self.build_block_parser()
        self.inlinePatterns = self.build_inlinepatterns()
        self.treeprocessors = self.build_treeprocessors()
        self.postprocessors = self.build_postprocessors()
        self.handle_zephyr_mirror()
        return self

    def build_preprocessors(self) -> markdown.util.Registry[markdown.preprocessors.Preprocessor]:
        preprocessors: markdown.util.Registry[markdown.preprocessors.Preprocessor] = markdown.util.Registry()
        preprocessors.register(MarkdownListPreprocessor(self), "hanging_lists", 35)
        preprocessors.register(markdown.preprocessors.NormalizeWhitespace(self), "normalize_whitespace", 30)
        preprocessors.register(fenced_code.FencedBlockPreprocessor(self), "fenced_code_block", 25)
        preprocessors.register(AlertWordNotificationProcessor(self), "custom_text_notifications", 20)
        return preprocessors

    def build_block_parser(self) -> BlockParser:
        parser: BlockParser = BlockParser(self)
        parser.blockprocessors.register(markdown.blockprocessors.EmptyBlockProcessor(parser), "empty", 95)
        parser.blockprocessors.register(ListIndentProcessor(parser), "indent", 90)
        if not self.email_gateway:
            parser.blockprocessors.register(markdown.blockprocessors.CodeBlockProcessor(parser), "code", 85)
        parser.blockprocessors.register(HashHeaderProcessor(parser), "hashheader", 80)
        parser.blockprocessors.register(markdown.blockprocessors.HRProcessor(parser), "hr", 70)
        parser.blockprocessors.register(OListProcessor(parser), "olist", 65)
        parser.blockprocessors.register(UListProcessor(parser), "ulist", 60)
        parser.blockprocessors.register(BlockQuoteProcessor(parser), "quote", 55)
        parser.blockprocessors.register(markdown.blockprocessors.ParagraphProcessor(parser), "paragraph", 50)
        return parser

    def build_inlinepatterns(self) -> markdown.util.Registry[markdown.inlinepatterns.Pattern]:
        DEL_RE: str = r"(?<!~)(\~\~)([^~\n]+?)(\~\~)(?!~)"
        EMPHASIS_RE: str = r"(\*)(?!\s+)([^\*^\n]+)(?<!\s)\*"
        STRONG_RE: str = r"(\*\*)([^\n]+?)\2"
        STRONG_EM_RE: str = r"(\*\*\*)(?!\s+)([^\*^\n]+)(?<!\s)\*\*\*"
        TEX_RE: str = r"\B(?<!\$)\$\$(?P<body>[^\n_$](\\\$|[^$\n])*)\$\$(?!\$)\B"
        TIMESTAMP_RE: str = r"<time:(?P<time>[^>]*?)>"
        reg: markdown.util.Registry[markdown.inlinepatterns.Pattern] = markdown.util.Registry()
        reg.register(BacktickInlineProcessor(markdown.inlinepatterns.BACKTICK_RE), "backtick", 105)
        reg.register(markdown.inlinepatterns.DoubleTagPattern(STRONG_EM_RE, "strong,em"), "strong_em", 100)
        reg.register(UserMentionPattern(mention.MENTIONS_RE, self), "usermention", 95)
        reg.register(Tex(TEX_RE, self), "tex", 90)
        reg.register(StreamTopicMessagePattern(get_compiled_stream_topic_message_link_regex(), self), "stream_topic_message", 89)
        reg.register(StreamTopicPattern(get_compiled_stream_topic_link_regex(), self), "topic", 87)
        reg.register(StreamPattern(get_compiled_stream_link_regex(), self), "stream", 85)
        reg.register(Timestamp(TIMESTAMP_RE), "timestamp", 75)
        reg.register(UserGroupMentionPattern(mention.USER_GROUP_MENTIONS_RE, self), "usergroupmention", 65)
        reg.register(LinkInlineProcessor(markdown.inlinepatterns.LINK_RE, self), "link", 60)
        reg.register(AutoLink(get_web_link_regex(), self), "autolink", 55)
        reg = self.register_linkifiers(reg)
        reg.register(markdown.inlinepatterns.HtmlInlineProcessor(markdown.inlinepatterns.ENTITY_RE, self), "entity", 40)
        reg.register(markdown.inlinepatterns.SimpleTagPattern(STRONG_RE, "strong"), "strong", 35)
        reg.register(markdown.inlinepatterns.SimpleTagPattern(EMPHASIS_RE, "em"), "emphasis", 30)
        reg.register(markdown.inlinepatterns.SimpleTagPattern(DEL_RE, "del"), "del", 25)
        reg.register(markdown.inlinepatterns.SimpleTextInlineProcessor(markdown.inlinepatterns.NOT_STRONG_RE), "not_strong", 20)
        reg.register(Emoji(EMOJI_REGEX, self), "emoji", 15)
        reg.register(EmoticonTranslation(EMOTICON_RE, self), "translate_emoticons", 10)
        reg.register(UnicodeEmoji(cast(Pattern[str], POSSIBLE_EMOJI_RE), self), "unicodeemoji", 0)
        return reg

    def register_linkifiers(self, registry: markdown.util.Registry[markdown.inlinepatterns.Pattern]) -> markdown.util.Registry[markdown.inlinepatterns.Pattern]:
        for linkifier in self.linkifiers:
            pattern: str = linkifier["pattern"]
            registry.register(LinkifierPattern(pattern, linkifier["url_template"], self), f"linkifiers/{pattern}", 45)
        return registry

    def build_treeprocessors(self) -> markdown.util.Registry[markdown.treeprocessors.Treeprocessor]:
        treeprocessors: markdown.util.Registry[markdown.treeprocessors.Treeprocessor] = markdown.util.Registry()
        treeprocessors.register(markdown.treeprocessors.InlineProcessor(self), "inline", 25)
        treeprocessors.register(markdown.treeprocessors.PrettifyTreeprocessor(self), "prettify", 20)
        treeprocessors.register(markdown.treeprocessors.UnescapeTreeprocessor(self), "unescape", 18)
        treeprocessors.register(InlineInterestingLinkProcessor(self), "inline_interesting_links", 15)
        if settings.CAMO_URI:
            treeprocessors.register(InlineImageProcessor(self), "rewrite_images_proxy", 10)
            treeprocessors.register(InlineVideoProcessor(self), "rewrite_videos_proxy", 10)
        return treeprocessors

    def build_postprocessors(self) -> markdown.util.Registry[markdown.postprocessors.Postprocessor]:
        postprocessors: markdown.util.Registry[markdown.postprocessors.Postprocessor] = markdown.util.Registry()
        postprocessors.register(markdown.postprocessors.RawHtmlPostprocessor(self), "raw_html", 20)
        postprocessors.register(markdown.postprocessors.AndSubstitutePostprocessor(), "amp_substitute", 15)
        return postprocessors

    def handle_zephyr_mirror(self) -> None:
        if self.linkifiers_key == ZEPHYR_MIRROR_MARKDOWN_KEY:
            self.inlinePatterns = get_sub_registry(self.inlinePatterns, ["autolink"])
            self.treeprocessors = get_sub_registry(self.treeprocessors, ["inline_interesting_links", "rewrite_images_proxy"])
            self.treeprocessors.register(markdown.treeprocessors.InlineProcessor(self), "inline", 25)
            self.preprocessors = get_sub_registry(self.preprocessors, ["custom_text_notifications"])
            self.parser.blockprocessors = get_sub_registry(self.parser.blockprocessors, ["paragraph"])


md_engines: dict[tuple[int, bool], ZulipMarkdown] = {}
linkifier_data: dict[int, list[LinkifierDict]] = {}


def make_md_engine(linkifiers_key: int, email_gateway: bool) -> None:
    md_engine_key: tuple[int, bool] = (linkifiers_key, email_gateway)
    if md_engine_key in md_engines:
        del md_engines[md_engine_key]
    linkifiers: list[LinkifierDict] = linkifier_data[linkifiers_key]
    md_engines[md_engine_key] = ZulipMarkdown(linkifiers=linkifiers, linkifiers_key=linkifiers_key, email_gateway=email_gateway)


basic_link_splitter: Pattern[str] = re.compile(r"[ !;\),\'\"]")


def percent_escape_format_string(format_string: str) -> str:
    return re.sub(r"(?<!%)(%%)*%([a-fA-F0-9][a-fA-F0-9])", r"\1%%\2", format_string)


@dataclass
class TopicLinkMatch:
    url: str
    text: str
    index: int
    precedence: Optional[int]


def topic_links(linkifiers_key: int, topic_name: str) -> list[dict[str, str]]:
    matches: list[TopicLinkMatch] = []
    linkifiers: list[LinkifierDict] = linkifiers_for_realm(linkifiers_key)
    precedence: int = 0
    options = re2.Options()
    options.log_errors = False
    for linkifier in linkifiers:
        raw_pattern: str = linkifier["pattern"]
        prepared_url_template = uri_template.URITemplate(linkifier["url_template"])
        try:
            pattern: Pattern[str] = re2.compile(prepare_linkifier_pattern(raw_pattern), options=options)
        except re2.error:
            continue
        pos: int = 0
        while pos < len(topic_name):
            m = pattern.search(topic_name, pos)
            if m is None:
                break
            match_details: dict[str, str] = m.groupdict()
            match_text: str = match_details[OUTER_CAPTURE_GROUP]
            pos = m.end() - len(match_details[AFTER_CAPTURE_GROUP])
            matches += [
                TopicLinkMatch(
                    url=prepared_url_template.expand(**match_details),
                    text=match_text,
                    index=m.start(),
                    precedence=precedence,
                )
            ]
        precedence += 1
    matches.sort(key=lambda k: (k.precedence, k.index))
    pos = 0
    while pos < len(topic_name):
        next_split = basic_link_splitter.search(topic_name, pos)
        end: int = next_split.start() if next_split is not None else len(topic_name)
        link_match = re.match(get_web_link_regex(), topic_name[pos:end])
        if link_match:
            actual_match_url: str = link_match.group("url")
            result = urlsplit(actual_match_url)
            if not result.scheme:
                if not result.netloc:
                    i = (result.path + "/").index("/")
                    result = result._replace(netloc=result.path[:i], path=result.path[i:])
                url_val = result._replace(scheme="https").geturl()
            else:
                url_val = actual_match_url
            matches.append(
                TopicLinkMatch(
                    url=url_val,
                    text=actual_match_url,
                    index=pos,
                    precedence=None,
                )
            )
        pos = end + 1

    def are_matches_overlapping(match_a: TopicLinkMatch, match_b: TopicLinkMatch) -> bool:
        return (match_b.index <= match_a.index < match_b.index + len(match_b.text)) or (match_a.index <= match_b.index < match_a.index + len(match_a.text))

    applied_matches: list[TopicLinkMatch] = []
    for current_match in matches:
        if all(not are_matches_overlapping(old_match, current_match) for old_match in applied_matches):
            applied_matches.append(current_match)
    applied_matches.sort(key=lambda v: v.index)
    return [{"url": match.url, "text": match.text} for match in applied_matches]


def maybe_update_markdown_engines(linkifiers_key: int, email_gateway: bool) -> None:
    linkifiers: list[LinkifierDict] = linkifiers_for_realm(linkifiers_key)
    if linkifiers_key not in linkifier_data or linkifier_data[linkifiers_key] != linkifiers:
        linkifier_data[linkifiers_key] = linkifiers
        for email_gateway_flag in [True, False]:
            if (linkifiers_key, email_gateway_flag) in md_engines:
                make_md_engine(linkifiers_key, email_gateway_flag)
    if (linkifiers_key, email_gateway) not in md_engines:
        make_md_engine(linkifiers_key, email_gateway)


_privacy_re: Pattern[str] = re.compile(r"\w")


def privacy_clean_markdown(content: str) -> str:
    return repr(_privacy_re.sub("x", content))


def do_convert(
    content: str,
    realm_alert_words_automaton: Optional[ahocorasick.Automaton] = None,
    message: Optional[Message] = None,
    message_realm: Optional[Realm] = None,
    sent_by_bot: bool = False,
    translate_emoticons: bool = False,
    url_embed_data: Optional[dict[str, Optional[UrlEmbedData]]] = None,
    mention_data: Optional[MentionData] = None,
    email_gateway: bool = False,
    no_previews: bool = False,
    acting_user: Optional[UserProfile] = None,
) -> MessageRenderingResult:
    if message is not None and message_realm is None:
        message_realm = message.get_realm()
    if message_realm is None:
        linkifiers_key: int = DEFAULT_MARKDOWN_KEY
    else:
        linkifiers_key = message_realm.id
    if message and hasattr(message, "id") and message.id:
        logging_message_id: str = "id# " + str(message.id)
    else:
        logging_message_id = "unknown"
    if (
        message is not None
        and message_realm is not None
        and message_realm.is_zephyr_mirror_realm
        and message.sending_client.name == "zephyr_mirror"
    ):
        linkifiers_key = ZEPHYR_MIRROR_MARKDOWN_KEY
    maybe_update_markdown_engines(linkifiers_key, email_gateway)
    md_engine_key: tuple[int, bool] = (linkifiers_key, email_gateway)
    _md_engine: ZulipMarkdown = md_engines[md_engine_key]
    _md_engine.reset()
    rendering_result: MessageRenderingResult = MessageRenderingResult(
        rendered_content="",
        mentions_topic_wildcard=False,
        mentions_stream_wildcard=False,
        mentions_user_ids=set(),
        mentions_user_group_ids=set(),
        alert_words=set(),
        links_for_preview=set(),
        user_ids_with_alert_words=set(),
        potential_attachment_path_ids=[],
        thumbnail_spinners=set(),
    )
    _md_engine.zulip_message = message
    _md_engine.zulip_rendering_result = rendering_result
    _md_engine.zulip_realm = message_realm
    _md_engine.zulip_db_data = None
    _md_engine.image_preview_enabled = image_preview_enabled(message, message_realm, no_previews)
    _md_engine.url_embed_preview_enabled = url_embed_preview_enabled(message, message_realm, no_previews)
    _md_engine.url_embed_data = url_embed_data
    user_upload_previews: Optional[dict[str, MarkdownImageMetadata]] = None
    if message_realm is not None:
        message_sender: Optional[UserProfile] = None
        if message is not None:
            message_sender = message.sender
        if mention_data is None:
            mention_backend = MentionBackend(message_realm.id)
            mention_data = MentionData(mention_backend, content, message_sender)
        if acting_user is None:
            acting_user = message_sender
        stream_names: set[str] = possible_linked_stream_names(content)
        stream_name_info: dict[str, int] = mention_data.get_stream_name_map(stream_names, acting_user=acting_user)
        linked_stream_topic_data: set[ChannelTopicInfo] = possible_linked_topics(content)
        topic_info: dict[ChannelTopicInfo, Optional[int]] = mention_data.get_topic_info_map(linked_stream_topic_data, acting_user=acting_user)
        if content_has_emoji_syntax(content):
            active_realm_emoji: dict[str, EmojiInfo] = get_name_keyed_dict_for_active_realm_emoji(message_realm.id)
        else:
            active_realm_emoji = {}
        user_upload_previews = get_user_upload_previews(message_realm.id, content)
        _md_engine.zulip_db_data = DbData(
            realm_alert_words_automaton=realm_alert_words_automaton,
            mention_data=mention_data,
            active_realm_emoji=active_realm_emoji,
            realm_url=message_realm.url,
            sent_by_bot=sent_by_bot,
            stream_names=stream_name_info,
            topic_info=topic_info,
            translate_emoticons=translate_emoticons,
            user_upload_previews=user_upload_previews,
        )
    try:
        rendering_result.rendered_content = unsafe_timeout(5, lambda: _md_engine.convert(content))
        if user_upload_previews is not None:
            content_with_thumbnails, thumbnail_spinners = rewrite_thumbnailed_images(rendering_result.rendered_content, user_upload_previews)
            rendering_result.thumbnail_spinners = thumbnail_spinners
            if content_with_thumbnails is not None:
                rendering_result.rendered_content = content_with_thumbnails
        MAX_MESSAGE_LENGTH: int = settings.MAX_MESSAGE_LENGTH
        if len(rendering_result.rendered_content) > MAX_MESSAGE_LENGTH * 100:
            raise MarkdownRenderingError(
                f"Rendered content exceeds {MAX_MESSAGE_LENGTH * 100} characters (message {logging_message_id})"
            )
        return rendering_result
    except Exception:
        cleaned: str = privacy_clean_markdown(content)
        markdown_logger.exception(
            "Exception in Markdown parser; input (sanitized) was: %s\n (message %s)",
            cleaned,
            logging_message_id,
        )
        raise MarkdownRenderingError
    finally:
        _md_engine.zulip_message = None
        _md_engine.zulip_realm = None
        _md_engine.zulip_db_data = None


markdown_time_start: float = 0.0
markdown_total_time: float = 0.0
markdown_total_requests: int = 0


def get_markdown_time() -> float:
    return markdown_total_time


def get_markdown_requests() -> int:
    return markdown_total_requests


def markdown_stats_start() -> None:
    global markdown_time_start
    markdown_time_start = time.time()


def markdown_stats_finish() -> None:
    global markdown_total_time, markdown_total_requests
    markdown_total_requests += 1
    markdown_total_time += time.time() - markdown_time_start


def markdown_convert(
    content: str,
    realm_alert_words_automaton: Optional[ahocorasick.Automaton] = None,
    message: Optional[Message] = None,
    message_realm: Optional[Realm] = None,
    sent_by_bot: bool = False,
    translate_emoticons: bool = False,
    url_embed_data: Optional[dict[str, Optional[UrlEmbedData]]] = None,
    mention_data: Optional[MentionData] = None,
    email_gateway: bool = False,
    no_previews: bool = False,
    acting_user: Optional[UserProfile] = None,
) -> MessageRenderingResult:
    markdown_stats_start()
    ret: MessageRenderingResult = do_convert(
        content,
        realm_alert_words_automaton,
        message,
        message_realm,
        sent_by_bot,
        translate_emoticons,
        url_embed_data,
        mention_data,
        email_gateway,
        no_previews=no_previews,
        acting_user=acting_user,
    )
    markdown_stats_finish()
    return ret


def render_message_markdown(
    message: Message,
    content: str,
    realm: Optional[Realm] = None,
    realm_alert_words_automaton: Optional[ahocorasick.Automaton] = None,
    url_embed_data: Optional[dict[str, Optional[UrlEmbedData]]] = None,
    mention_data: Optional[MentionData] = None,
    email_gateway: bool = False,
    acting_user: Optional[UserProfile] = None,
) -> MessageRenderingResult:
    if realm is None:
        realm = message.get_realm()
    sender: UserProfile = message.sender
    sent_by_bot: bool = sender.is_bot
    translate_emoticons: bool = sender.translate_emoticons
    rendering_result: MessageRenderingResult = markdown_convert(
        content,
        realm_alert_words_automaton=realm_alert_words_automaton,
        message=message,
        message_realm=realm,
        sent_by_bot=sent_by_bot,
        translate_emoticons=translate_emoticons,
        url_embed_data=url_embed_data,
        mention_data=mention_data,
        email_gateway=email_gateway,
        acting_user=acting_user,
    )
    return rendering_result