import getpass
import json
import os
import platform
import re
import shutil
import string
import sys
import tempfile
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, IO, Iterator, List, Optional, Set, Union, cast, Dict, Tuple, ContextManager
from urllib.parse import urlparse

import requests
import urllib3  # type: ignore

from .exceptions import *
from .instaloadercontext import InstaloaderContext, RateController
from .lateststamps import LatestStamps
from .nodeiterator import NodeIterator, resumable_iteration
from .sectioniterator import SectionIterator
from .structures import (Hashtag, Highlight, JsonExportable, Post, PostLocation, Profile, Story, StoryItem,
                         load_structure_from_file, save_structure_to_file, PostSidecarNode, TitlePic)


def _get_config_dir() -> str:
    if platform.system() == "Windows":
        localappdata = os.getenv("LOCALAPPDATA")
        if localappdata is not None:
            return os.path.join(localappdata, "Instaloader")
        return os.path.join(tempfile.gettempdir(), ".instaloader-" + getpass.getuser())
    return os.path.join(os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "instaloader")


def get_default_session_filename(username: str) -> str:
    return os.path.join(_get_config_dir(), "session-{}".format(username))


def get_legacy_session_filename(username: str) -> str:
    dirname = tempfile.gettempdir() + "/" + ".instaloader-" + getpass.getuser()
    filename = dirname + "/" + "session-" + username
    return filename.lower()


def get_default_stamps_filename() -> str:
    configdir = _get_config_dir()
    return os.path.join(configdir, "latest-stamps.ini")


def format_string_contains_key(format_string: str, key: str) -> bool:
    for literal_text, field_name, format_spec, conversion in string.Formatter().parse(format_string):
        if field_name and (field_name == key or field_name.startswith(key + '.')):
            return True
    return False


def _requires_login(func: Callable) -> Callable:
    @wraps(func)
    def call(instaloader: 'Instaloader', *args: Any, **kwargs: Any) -> Any:
        if not instaloader.context.is_logged_in:
            raise LoginRequiredException("Login required.")
        return func(instaloader, *args, **kwargs)
    return call


def _retry_on_connection_error(func: Callable) -> Callable:
    @wraps(func)
    def call(instaloader: 'Instaloader', *args: Any, **kwargs: Any) -> Any:
        try:
            return func(instaloader, *args, **kwargs)
        except (urllib3.exceptions.HTTPError, requests.exceptions.RequestException, ConnectionException) as err:
            error_string = "{}({}): {}".format(func.__name__, ', '.join([repr(arg) for arg in args]), err)
            if (kwargs.get('_attempt') or 1) == instaloader.context.max_connection_attempts:
                raise ConnectionException(error_string) from None
            instaloader.context.error(error_string + " [retrying; skip with ^C]", repeat_at_end=False)
            try:
                if kwargs.get('_attempt'):
                    kwargs['_attempt'] += 1
                else:
                    kwargs['_attempt'] = 2
                instaloader.context.do_sleep()
                return call(instaloader, *args, **kwargs)
            except KeyboardInterrupt:
                instaloader.context.error("[skipped by user]", repeat_at_end=False)
                raise ConnectionException(error_string) from None
    return call


class _ArbitraryItemFormatter(string.Formatter):
    def __init__(self, item: Any):
        self._item = item

    def get_value(self, key: str, args: Any, kwargs: Any) -> Any:
        if key == 'filename' and isinstance(self._item, (Post, StoryItem, PostSidecarNode, TitlePic)):
            return "{filename}"
        if hasattr(self._item, key):
            return getattr(self._item, key)
        return super().get_value(key, args, kwargs)

    def format_field(self, value: Any, format_spec: str) -> str:
        if isinstance(value, datetime) and not format_spec:
            return super().format_field(value, '%Y-%m-%d_%H-%M-%S')
        if value is None:
            return ''
        return super().format_field(value, format_spec)


class _PostPathFormatter(_ArbitraryItemFormatter):
    RESERVED: Set[str] = {'CON', 'PRN', 'AUX', 'NUL',
                     'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                     'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}

    def __init__(self, item: Any, force_windows_path: bool = False):
        super().__init__(item)
        self.force_windows_path = force_windows_path

    def get_value(self, key: str, args: Any, kwargs: Any) -> Any:
        ret = super().get_value(key, args, kwargs)
        if not isinstance(ret, str):
            return ret
        return self.sanitize_path(ret, self.force_windows_path)

    @staticmethod
    def sanitize_path(ret: str, force_windows_path: bool = False) -> str:
        ret = ret.replace('/', '\u2215')
        if ret.startswith('.'):
            ret = ret.replace('.', '\u2024', 1)
        if force_windows_path or platform.system() == 'Windows':
            ret = ret.replace(':', '\uff1a').replace('<', '\ufe64').replace('>', '\ufe65').replace('\"', '\uff02')
            ret = ret.replace('\\', '\ufe68').replace('|', '\uff5c').replace('?', '\ufe16').replace('*', '\uff0a')
            ret = ret.replace('\n', ' ').replace('\r', ' ')
            root, ext = os.path.splitext(ret)
            if root.upper() in _PostPathFormatter.RESERVED:
                root += '_'
            if ext == '.':
                ext = '\u2024'
            ret = root + ext
        return ret


class Instaloader:
    def __init__(self,
                 sleep: bool = True,
                 quiet: bool = False,
                 user_agent: Optional[str] = None,
                 dirname_pattern: Optional[str] = None,
                 filename_pattern: Optional[str] = None,
                 download_pictures: bool = True,
                 download_videos: bool = True,
                 download_video_thumbnails: bool = True,
                 download_geotags: bool = False,
                 download_comments: bool = False,
                 save_metadata: bool = True,
                 compress_json: bool = True,
                 post_metadata_txt_pattern: Optional[str] = None,
                 storyitem_metadata_txt_pattern: Optional[str] = None,
                 max_connection_attempts: int = 3,
                 request_timeout: float = 300.0,
                 rate_controller: Optional[Callable[[InstaloaderContext], RateController]] = None,
                 resume_prefix: Optional[str] = "iterator",
                 check_resume_bbd: bool = True,
                 slide: Optional[str] = None,
                 fatal_status_codes: Optional[List[int]] = None,
                 iphone_support: bool = True,
                 title_pattern: Optional[str] = None,
                 sanitize_paths: bool = False):

        self.context = InstaloaderContext(sleep, quiet, user_agent, max_connection_attempts,
                                          request_timeout, rate_controller, fatal_status_codes,
                                          iphone_support)

        self.dirname_pattern = dirname_pattern or "{target}"
        self.filename_pattern = filename_pattern or "{date_utc}_UTC"
        if title_pattern is not None:
            self.title_pattern = title_pattern
        else:
            if (format_string_contains_key(self.dirname_pattern, 'profile') or
                format_string_contains_key(self.dirname_pattern, 'target')):
                self.title_pattern = '{date_utc}_UTC_{typename}'
            else:
                self.title_pattern = '{target}_{date_utc}_UTC_{typename}'
        self.sanitize_paths = sanitize_paths
        self.download_pictures = download_pictures
        self.download_videos = download_videos
        self.download_video_thumbnails = download_video_thumbnails
        self.download_geotags = download_geotags
        self.download_comments = download_comments
        self.save_metadata = save_metadata
        self.compress_json = compress_json
        self.post_metadata_txt_pattern = '{caption}' if post_metadata_txt_pattern is None \
            else post_metadata_txt_pattern
        self.storyitem_metadata_txt_pattern = '' if storyitem_metadata_txt_pattern is None \
            else storyitem_metadata_txt_pattern
        self.resume_prefix = resume_prefix
        self.check_resume_bbd = check_resume_bbd

        self.slide = slide or ""
        self.slide_start = 0
        self.slide_end = -1
        if self.slide != "":
            splitted = self.slide.split('-')
            if len(splitted) == 1:
                if splitted[0] == 'last':
                    self.slide_start = -1
                else:
                    if int(splitted[0]) > 0:
                        self.slide_start = self.slide_end = int(splitted[0])-1
                    else:
                        raise InvalidArgumentException("--slide parameter must be greater than 0.")
            elif len(splitted) == 2:
                if splitted[1] == 'last':
                    self.slide_start = int(splitted[0])-1
                elif 0 < int(splitted[0]) < int(splitted[1]):
                    self.slide_start = int(splitted[0])-1
                    self.slide_end = int(splitted[1])-1
                else:
                    raise InvalidArgumentException("Invalid data for --slide parameter.")
            else:
                raise InvalidArgumentException("Invalid data for --slide parameter.")

    @contextmanager
    def anonymous_copy(self) -> Iterator['Instaloader']:
        new_loader = Instaloader(
            sleep=self.context.sleep,
            quiet=self.context.quiet,
            user_agent=self.context.user_agent,
            dirname_pattern=self.dirname_pattern,
            filename_pattern=self.filename_pattern,
            download_pictures=self.download_pictures,
            download_videos=self.download_videos,
            download_video_thumbnails=self.download_video_thumbnails,
            download_geotags=self.download_geotags,
            download_comments=self.download_comments,
            save_metadata=self.save_metadata,
            compress_json=self.compress_json,
            post_metadata_txt_pattern=self.post_metadata_txt_pattern,
            storyitem_metadata_txt_pattern=self.storyitem_metadata_txt_pattern,
            max_connection_attempts=self.context.max_connection_attempts,
            request_timeout=self.context.request_timeout,
            resume_prefix=self.resume_prefix,
            check_resume_bbd=self.check_resume_bbd,
            slide=self.slide,
            fatal_status_codes=self.context.fatal_status_codes,
            iphone_support=self.context.iphone_support,
            sanitize_paths=self.sanitize_paths)
        yield new_loader
        self.context.error_log.extend(new_loader.context.error_log)
        new_loader.context.error_log = []
        new_loader.close()

    def close(self) -> None:
        self.context.close()

    def __enter__(self) -> 'Instaloader':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @_retry_on_connection_error
    def download_pic(self, filename: str, url: str, mtime: datetime,
                     filename_suffix: Optional[str] = None, _attempt: int = 1) -> bool:
        if filename_suffix is not None:
            filename += '_' + filename_suffix
        urlmatch = re.search('\\.[a-z0-9]*\\?', url)
        file_extension = url[-3:] if urlmatch is None else urlmatch.group(0)[1:-1]
        nominal_filename = filename + '.' + file_extension
        if os.path.isfile(nominal_filename):
            self.context.log(nominal_filename + ' exists', end=' ', flush=True)
            return False
        resp = self.context.get_raw(url)
        if 'Content-Type' in resp.headers and resp.headers['Content-Type']:
            header_extension = '.' + resp.headers['Content-Type'].split(';')[0].split('/')[-1]
            header_extension = header_extension.lower().replace('jpeg', 'jpg')
            filename += header_extension
        else:
            filename = nominal_filename
        if filename != nominal_filename and os.path.isfile(filename):
            self.context.log(filename + ' exists', end=' ', flush=True)
            return False
        self.context.write_raw(resp, filename)
        os.utime(filename, (datetime.now().timestamp(), mtime.timestamp()))
        return True

    def save_metadata_json(self, filename: str, structure: JsonExportable) -> None:
        if self.compress_json:
            filename += '.json.xz'
        else:
            filename += '.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_structure_to_file(structure, filename)
        if isinstance(structure, (Post, StoryItem)):
            self.context.log('json', end=' ', flush=True)

    def update_comments(self, filename: str, post: Post) -> None:
        def _postcommentanswer_asdict(comment: Any) -> Dict[str, Any]:
            return {'id': comment.id,
                    'created_at': int(comment.created_at_utc.replace(tzinfo=timezone.utc).timestamp()),
                    'text': comment.text,
                    'owner': comment.owner._asdict(),
                    'likes_count': comment.likes_count}

        def _postcomment_asdict(comment: Any) -> Dict[str, Any]:
            return {**_postcommentanswer_asdict(comment),
                    'answers': sorted([_postcommentanswer_asdict(answer) for answer in comment.answers],
                                      key=lambda t: int(t['id']),
                                      reverse=True)}

        def get_unique_comments(comments: List[Dict[str, Any]], combine_answers: bool = False) -> List[Dict[str, Any]]:
            if not comments:
                return list()
            comments_list = sorted(sorted(list(comments), key=lambda t: int(t['id'])),
                                   key=lambda t: int(t['created_at']), reverse=True)
            unique_comments_list = [comments_list[0]]
            for x, y in zip(comments_list[:-1], comments_list[1:]):
                if x['id'] != y['id']:
                    unique_comments_list.append(y)
                else:
                    unique_comments_list[-1]['likes_count'] = y.get('likes_count')
                    if combine_answers:
                        combined_answers = unique_comments_list[-1].get('answers') or list()
                        if 'answers' in y:
                            combined_answers.extend(y['answers'])
                        unique_comments_list[-1]['answers'] = get_unique_comments(combined_answers)
            return unique_comments_list

        def get_new_comments(new_comments: Iterator[Any], start: int) -> Iterator[Dict[str, Any]]:
            for idx, comment in enumerate(new_comments, start=start+1):
                if idx % 250 == 0:
                    self.context.log('{}'.format(idx), end='…', flush=True)
                yield comment

        def save_comments(extended_comments: List[Dict[str, Any]]) -> None:
            unique_comments = get_unique_comments(extended_comments, combine_answers=True)
            answer_ids = set(int(answer['id']) for comment in unique_comments for answer in comment.get('answers', []))
            with open(filename, 'w') as file:
                file.write(json.dumps(list(filter(lambda t: int(t['id']) not in answer_ids, unique_comments)),
                                      indent=4))

        base_filename = filename
        filename += '_comments.json'
        try:
            with open(filename) as fp:
                comments = json.load(fp)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            comments = list()

        comments_iterator = post.get_comments()
        try:
            with resumable_iteration(
                    context=self.context,
                    iterator=comments_iterator,
                    load=load_structure_from_file,
                    save=save_structure_to_file,
                    format_path=lambda magic: "{}_{}_{}.json.xz".format(base_filename, self.resume_prefix, magic),
                    check_bbd=self.check_resume_bbd,
                    enabled=self.resume_prefix is not None
            ) as (_is_resuming, start_index):
                comments.extend(_postcomment_asdict(comment)
                                for comment in get_new_comments(comments_iterator, start_index))
        except (KeyboardInterrupt, AbortDownloadException):
            if comments:
                save_comments(comments)
            raise
        if comments:
            save_comments(comments)
            self.context.log('comments', end=' ', flush=True)

    def save_caption(self, filename: str, mtime: datetime, caption: str) -> None:
        def _elliptify(caption: str) -> str:
            pcaption = caption.replace('\n', ' ').strip()
            return '[' + ((pcaption[:29] + "\u2026") if len(pcaption) > 31 else pcaption) + ']'
        filename += '.txt'
        caption += '\n'
        pcaption = _elliptify(caption)
