from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, TypeVar, Union, cast
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
from zerver.lib.mention import BEFORE_MENTION_ALLOWED_REGEX, ChannelTopicInfo, FullNameInfo, MentionBackend, MentionData, get_user_group_mention_display_name
from zerver.lib.outgoing_http import OutgoingSession
from zerver.lib.subdomains import is_static_or_current_realm_url
from zerver.lib.tex import render_tex
from zerver.lib.thumbnail import MarkdownImageMetadata, get_user_upload_previews, rewrite_thumbnailed_images
from zerver.lib.timeout import unsafe_timeout
from zerver.lib.timezone import common_timezones
from zerver.lib.types import LinkifierDict
from zerver.lib.url_encoding import encode_stream, hash_util_encode
from zerver.lib.url_preview.types import UrlEmbedData, UrlOEmbedData
from zerver.models import Message, Realm, UserProfile
from zerver.models.linkifiers import linkifiers_for_realm
from zerver.models.realm_emoji import EmojiInfo, get_name_keyed_dict_for_active_realm_emoji

ReturnT = TypeVar('ReturnT')
html_safelisted_schemes = ('bitcoin', 'geo', 'im', 'irc', 'ircs', 'magnet', 'mailto', 'matrix', 'mms', 'news', 'nntp', 'openpgp4fpr', 'sip', 'sms', 'smsto', 'ssh', 'tel', 'urn', 'webcal', 'wtai', 'xmpp')
allowed_schemes = ('http', 'https', 'ftp', 'file', *html_safelisted_schemes)

class LinkInfo(TypedDict):
    pass

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
    realm_alert_words_automaton: Optional[Any]
    mention_data: Optional[MentionData]
    active_realm_emoji: Dict[str, EmojiInfo]
    realm_url: str
    sent_by_bot: bool
    stream_names: Dict[str, int]
    topic_info: Dict[ChannelTopicInfo, str]
    translate_emoticons: bool
    user_upload_previews: Dict[str, MarkdownImageMetadata]

version: int = 1
_T = TypeVar('_T')
ElementStringNone = Union[Element, str, None]
EMOJI_REGEX: str = '(?P<syntax>:[\\w\\-\\+]+:)'

def verbose_compile(pattern: str) -> Pattern[str]:
    return re.compile(f'^(.*?){pattern}(.*?)$', re.DOTALL | re.VERBOSE)

STREAM_LINK_REGEX: str = f'\n                     {BEFORE_MENTION_ALLOWED_REGEX} # Start after whitespace or specified chars\n                     \\#\\*\\*                         # and after hash sign followed by double asterisks\n                         (?P<stream_name>[^\\*]+)    # stream name can contain anything\n                     \\*\\*                           # ends by double asterisks\n                    '

@lru_cache(None)
def get_compiled_stream_link_regex() -> Pattern[str]:
    return re.compile(STREAM_LINK_REGEX, re.DOTALL | re.VERBOSE)

STREAM_TOPIC_LINK_REGEX: str = f'\n                     {BEFORE_MENTION_ALLOWED_REGEX}  # Start after whitespace or specified chars\n                     \\#\\*\\*                          # and after hash sign followed by double asterisks\n                         (?P<stream_name>[^\\*>]+)    # stream name can contain anything except >\n                         >                           # > acts as separator\n                         (?P<topic_name>[^\\*]*)      # topic name can be an empty string or contain anything\n                     \\*\\*                            # ends by double asterisks\n                   '

@lru_cache(None)
def get_compiled_stream_topic_link_regex() -> Pattern[str]:
    return re.compile(STREAM_TOPIC_LINK_REGEX, re.DOTALL | re.VERBOSE)

STREAM_TOPIC_MESSAGE_LINK_REGEX: str = f'\n                     {BEFORE_MENTION_ALLOWED_REGEX}  # Start after whitespace or specified chars\n                     \\#\\*\\*                          # and after hash sign followed by double asterisks\n                         (?P<stream_name>[^\\*>]+)    # stream name can contain anything except >\n                         >                           # > acts as separator\n                         (?P<topic_name>[^\\*]*)      # topic name can be an empty string or contain anything\n                         @\n                         (?P<message_id>\\d+)         # message id\n                     \\*\\*                            # ends by double asterisks\n                   '

@lru_cache(None)
def get_compiled_stream_topic_message_link_regex() -> Pattern[str]:
    return re.compile(STREAM_TOPIC_MESSAGE_LINK_REGEX, re.DOTALL | re.VERBOSE)

@lru_cache(None)
def get_web_link_regex() -> Pattern[str]:
    tlds = '|'.join(list_of_tlds())
    inner_paren_contents = '[^\\s()\\"]*'
    paren_group = '\n                    [^\\s()\\"]*?            # Containing characters that won\'t end the URL\n                    (?: \\( %s \\)           # and more characters in matched parens\n                        [^\\s()\\"]*?        # followed by more characters\n                    )*                     # zero-or-more sets of paired parens\n                   '
    nested_paren_chunk = paren_group
    for i in range(6):
        nested_paren_chunk %= (paren_group,)
    nested_paren_chunk %= (inner_paren_contents,)
    file_links = '| (?:file://(/[^/ ]*)+/?)' if settings.ENABLE_FILE_LINKS else ''
    REGEX = f"""\n        (?<![^\\s'"\\(,:<])    # Start after whitespace or specified chars\n                             # (Double-negative lookbehind to allow start-of-string)\n        (?P<url>             # Main group\n            (?:(?:           # Domain part\n                https?://[\\w.:@-]+?   # If it has a protocol, anything goes.\n               |(?:                   # Or, if not, be more strict to avoid false-positives\n                    (?:[\\w-]+\\.)+     # One or more domain components, separated by dots\n                    (?:{tlds})        # TLDs\n                )\n            )\n            (?:/             # A path, beginning with /\n                {nested_paren_chunk}           # zero-to-6 sets of paired parens\n            )?)              # Path is optional\n            | (?:[\\w.-]+\\@[\\w.-]+\\.[\\w]+) # Email is separate, since it can't have a path\n            {file_links}               # File path start with file:///, enable by setting ENABLE_FILE_LINKS=True\n            | (?:bitcoin:[13][a-km-zA-HJ-NP-Z1-9]{{25,34}})  # Bitcoin address pattern, see https://mokagio.github.io/tech-journal/2014/11/21/regex-bitcoin.html\n        )\n        (?=                            # URL must be followed by (not included in group)\n            [!:;\\?\\),\\.\\'\\"\\>]*         # Optional punctuation characters\n            (?:\\Z|\\s)                  # followed by whitespace or end of string\n        )\n        """
    return verbose_compile(REGEX)

def clear_web_link_regex_for_testing() -> None:
    get_web_link_regex.cache_clear()

markdown_logger: logging.Logger = logging.getLogger()

def rewrite_local_links_to_relative(db_data: Optional[DbData], link: str) -> str:
    """If the link points to a local destination (e.g. #narrow/...),
    generate a relative link that will open it in the current window.
    """
    if db_data:
        realm_url_prefix = db_data.realm_url + '/'
        if link.startswith((realm_url_prefix + '#', realm_url_prefix + 'user_uploads/')):
            return link.removeprefix(realm_url_prefix)
    return link

def url_embed_preview_enabled(message: Optional[Message] = None, realm: Optional[Realm] = None, no_previews: bool = False) -> bool:
    if not settings.INLINE_URL_EMBED_PREVIEW:
        return False
    if no_previews:
        return False
    if realm is None and message is not None:
        realm = message.get_realm()
    if realm is None:
        return True
    return realm.inline_url_embed_preview

def image_preview_enabled(message: Optional[Message] = None, realm: Optional[Realm] = None, no_previews: bool = False) -> bool:
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
    common_false_positives = {'java', 'md', 'mov', 'py', 'zip'}
    return sorted(tld_set - common_false_positives, key=len, reverse=True)

def walk_tree(root: Element, processor: Callable[[Element], Optional[T]], stop_after_first: bool = False) -> List[T]:
    results: List[T] = []
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

T = TypeVar('T')

class ResultWithFamily(Generic[T]):
    def __init__(self, family: ElementFamily, result: T) -> None:
        self.family = family
        self.result = result

class ElementPair:
    def __init__(self, parent: Optional['ElementPair'], value: Element) -> None:
        self.parent = parent
        self.value = value

def walk_tree_with_family(root: Element, processor: Callable[[Element], Optional[T]]) -> List[ResultWithFamily[T]]:
    results: List[ResultWithFamily[T]] = []
    queue = deque([ElementPair(parent=None, value=root)])
    while queue:
        currElementPair = queue.popleft()
        for child in currElementPair.value:
            queue.append(ElementPair(parent=currElementPair, value=child))
            result = processor(child)
            if result is not None:
                if currElementPair.parent is not None:
                    grandparent_element = currElementPair.parent
                    grandparent = grandparent_element.value
                else:
                    grandparent = None
                family = ElementFamily(grandparent=grandparent, parent=currElementPair.value, child=child, in_blockquote=has_blockquote_ancestor(currElementPair))
                results.append(ResultWithFamily(family=family, result=result))
    return results

def has_blockquote_ancestor(element_pair: Optional[ElementPair]) -> bool:
    if element_pair is None:
        return False
    elif element_pair.value.tag == 'blockquote':
        return True
    else:
        return has_blockquote_ancestor(element_pair.parent)

@cache_with_key(lambda tweet_id: tweet_id, cache_name='database')
def fetch_tweet_data(tweet_id: str) -> Optional[Dict[str, Any]]:
    raise NotImplementedError('Twitter desupported their v1 API')

class OpenGraphSession(OutgoingSession):
    def __init__(self) -> None:
        super().__init__(role='markdown', timeout=1)

def fetch_open_graph_image(url: str) -> Optional[Dict[str, Optional[str]]]:
    og = {'image': None, 'title': None, 'desc': None}
    try:
        with OpenGraphSession().get(url, headers={'Accept': 'text/html,application/xhtml+xml'}, stream=True) as res:
            if res.status_code != requests.codes.ok:
                return None
            m = EmailMessage()
            m['Content-Type'] = res.headers.get('Content-Type')
            mimetype = m.get_content_type()
            if mimetype not in ('text/html', 'application/xhtml+xml'):
                return None
            html = mimetype == 'text/html'
            res.raw.decode_content = True
            for event, element in lxml.etree.iterparse(res.raw, events=('start',), no_network=True, remove_comments=True, html=html):
                parent = element.getparent()
                if parent is not None:
                    parent.text = None
                    parent.remove(element)
                if element.tag in ('body', '{http://www.w3.org/1999/xhtml}body'):
                    break
                elif element.tag in ('meta', '{http://www.w3.org/1999/xhtml}meta'):
                    if element.get('property') == 'og:image':
                        content = element.get('content')
                        if content is not None:
                            og['image'] = urljoin(res.url, content)
                    elif element.get('property') == 'og:title':
                        og['title'] = element.get('content')
                    elif element.get('property') == 'og:description':
                        og['desc'] = element.get('content')
    except (requests.RequestException, urllib3.exceptions.HTTPError):
        return None
    return None if og['image'] is None else og

def get_tweet_id(url: str) -> Optional[str]:
    parsed_url = urlsplit(url)
    if not (parsed_url.netloc == 'twitter.com' or parsed_url.netloc.endswith('.twitter.com')):
        return None
    to_match = parsed_url.path
    if parsed_url.path == '/' and len(parsed_url.fragment) > 5:
        to_match = parsed_url.fragment
    tweet_id_match = re.match('^!?/.*?/status(es)?/(?P<tweetid>\\d{10,30})(/photo/[0-9])?/?$', to_match)
    if not tweet_id_match:
        return None
    return tweet_id_match.group('tweetid')

class InlineImageProcessor(markdown.treeprocessors.Treeprocessor):
    """
    Rewrite inline img tags to serve external content via Camo.

    This rewrites all images, except ones that are served from the current
    realm or global STATIC_URL. This is to ensure that each realm only loads
    images that are hosted on that realm or by the global installation,
    avoiding information leakage to external domains or between realms. We need
    to disable proxying of images hosted on the same realm, because otherwise
    we will break images in /user_uploads/, which require authorization to
    view.
    """

    def __init__(self, zmd: 'ZulipMarkdown') -> None:
        super().__init__(zmd)
        self.zmd = zmd

    @override
    def run(self, root: Element) -> Element:
        found_imgs = walk_tree(root, lambda e: e if e.tag == 'img' else None)
        for img in found_imgs:
            url = img.get('src')
            assert url is not None
            if is_static_or_current_realm_url(url, self.zmd.zulip_realm):
                continue
            img.set('src', get_camo_url(url))
        return root

class InlineVideoProcessor(markdown.treeprocessors.Treeprocessor):
    """
    Rewrite inline video tags to serve external content via Camo.

    This rewrites all video, except ones that are served from the current
    realm or global STATIC_URL. This is to ensure that each realm only loads
    videos that are hosted on that realm or by the global installation,
    avoiding information leakage to external domains or between realms. We need
    to disable proxying of videos hosted on the same realm, because otherwise
    we will break videos in /user_uploads/, which require authorization to
    view.
    """

    def __init