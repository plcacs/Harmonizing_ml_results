#!/usr/bin/env python3
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
from typing import Any, Callable, ContextManager, IO, Iterator, List, Optional, Set, Union, cast, Dict
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
        # on Windows, use %LOCALAPPDATA%\Instaloader
        localappdata: Optional[str] = os.getenv("LOCALAPPDATA")
        if localappdata is not None:
            return os.path.join(localappdata, "Instaloader")
        # legacy fallback - store in temp dir if %LOCALAPPDATA% is not set
        return os.path.join(tempfile.gettempdir(), ".instaloader-" + getpass.getuser())
    # on Unix, use ~/.config/instaloader
    return os.path.join(os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "instaloader")


def get_default_session_filename(username: str) -> str:
    """Returns default session filename for given username."""
    configdir: str = _get_config_dir()
    sessionfilename: str = "session-{}".format(username)
    return os.path.join(configdir, sessionfilename)


def get_legacy_session_filename(username: str) -> str:
    """Returns legacy (until v4.4.3) default session filename for given username."""
    dirname: str = tempfile.gettempdir() + "/" + ".instaloader-" + getpass.getuser()
    filename: str = dirname + "/" + "session-" + username
    return filename.lower()


def get_default_stamps_filename() -> str:
    """
    Returns default filename for latest stamps database.

    .. versionadded:: 4.8

    """
    configdir: str = _get_config_dir()
    return os.path.join(configdir, "latest-stamps.ini")


def format_string_contains_key(format_string: str, key: str) -> bool:
    for literal_text, field_name, format_spec, conversion in string.Formatter().parse(format_string):
        if field_name and (field_name == key or field_name.startswith(key + '.')):
            return True
    return False


def _requires_login(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to raise an exception if herewith-decorated function is called without being logged in"""
    @wraps(func)
    def call(instaloader: "Instaloader", *args: Any, **kwargs: Any) -> Any:
        if not instaloader.context.is_logged_in:
            raise LoginRequiredException("Login required.")
        return func(instaloader, *args, **kwargs)
    return call


def _retry_on_connection_error(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to retry the function max_connection_attempts number of times.

    Herewith-decorated functions need an ``_attempt`` keyword argument.
    """
    @wraps(func)
    def call(instaloader: "Instaloader", *args: Any, **kwargs: Any) -> Any:
        try:
            return func(instaloader, *args, **kwargs)
        except (urllib3.exceptions.HTTPError, requests.exceptions.RequestException, ConnectionException) as err:
            error_string: str = "{}({}): {}".format(func.__name__, ', '.join([repr(arg) for arg in args]), err)
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
    def __init__(self, item: Any) -> None:
        self._item: Any = item

    def get_value(self, key: Any, args: Any, kwargs: Any) -> Any:
        """Override to substitute {ATTRIBUTE} by attributes of our _item."""
        if key == 'filename' and isinstance(self._item, (Post, StoryItem, PostSidecarNode, TitlePic)):
            return "{filename}"
        if hasattr(self._item, key):
            return getattr(self._item, key)
        return super().get_value(key, args, kwargs)

    def format_field(self, value: Any, format_spec: str) -> str:
        """Override to have our default format_spec for datetime objects, and to let None yield an empty string."""
        if isinstance(value, datetime) and not format_spec:
            return super().format_field(value, '%Y-%m-%d_%H-%M-%S')
        if value is None:
            return ''
        return super().format_field(value, format_spec)


class _PostPathFormatter(_ArbitraryItemFormatter):
    RESERVED: Set[str] = {'CON', 'PRN', 'AUX', 'NUL',
                     'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                     'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}

    def __init__(self, item: Any, force_windows_path: bool = False) -> None:
        super().__init__(item)
        self.force_windows_path: bool = force_windows_path

    def get_value(self, key: Any, args: Any, kwargs: Any) -> Any:
        ret: Any = super().get_value(key, args, kwargs)
        if not isinstance(ret, str):
            return ret
        return self.sanitize_path(ret, self.force_windows_path)

    @staticmethod
    def sanitize_path(ret: str, force_windows_path: bool = False) -> str:
        """Replaces '/' with similar looking Division Slash and some other illegal filename characters on Windows."""
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
                 sanitize_paths: bool = False) -> None:

        self.context: InstaloaderContext = InstaloaderContext(sleep, quiet, user_agent, max_connection_attempts,
                                                              request_timeout, rate_controller, fatal_status_codes,
                                                              iphone_support)
        self.dirname_pattern: str = dirname_pattern or "{target}"
        self.filename_pattern: str = filename_pattern or "{date_utc}_UTC"
        if title_pattern is not None:
            self.title_pattern: str = title_pattern
        else:
            if (format_string_contains_key(self.dirname_pattern, 'profile') or
                format_string_contains_key(self.dirname_pattern, 'target')):
                self.title_pattern = '{date_utc}_UTC_{typename}'
            else:
                self.title_pattern = '{target}_{date_utc}_UTC_{typename}'
        self.sanitize_paths: bool = sanitize_paths
        self.download_pictures: bool = download_pictures
        self.download_videos: bool = download_videos
        self.download_video_thumbnails: bool = download_video_thumbnails
        self.download_geotags: bool = download_geotags
        self.download_comments: bool = download_comments
        self.save_metadata: bool = save_metadata
        self.compress_json: bool = compress_json
        self.post_metadata_txt_pattern: str = '{caption}' if post_metadata_txt_pattern is None \
            else post_metadata_txt_pattern
        self.storyitem_metadata_txt_pattern: str = '' if storyitem_metadata_txt_pattern is None \
            else storyitem_metadata_txt_pattern
        self.resume_prefix: Optional[str] = resume_prefix
        self.check_resume_bbd: bool = check_resume_bbd

        self.slide: str = slide or ""
        self.slide_start: int = 0
        self.slide_end: int = -1
        if self.slide != "":
            splitted: List[str] = self.slide.split('-')
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
    def anonymous_copy(self) -> Iterator["Instaloader"]:
        new_loader: Instaloader = Instaloader(
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
        try:
            yield new_loader
        finally:
            self.context.error_log.extend(new_loader.context.error_log)
            new_loader.context.error_log = []  # avoid double-printing of errors
            new_loader.close()

    def close(self) -> None:
        self.context.close()

    def __enter__(self) -> "Instaloader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @_retry_on_connection_error
    def download_pic(self, filename: str, url: str, mtime: datetime,
                     filename_suffix: Optional[str] = None, _attempt: int = 1) -> bool:
        if filename_suffix is not None:
            filename += '_' + filename_suffix
        urlmatch = re.search('\\.[a-z0-9]*\\?', url)
        file_extension: str = url[-3:] if urlmatch is None else urlmatch.group(0)[1:-1]
        nominal_filename: str = filename + '.' + file_extension
        if os.path.isfile(nominal_filename):
            self.context.log(nominal_filename + ' exists', end=' ', flush=True)
            return False
        resp: requests.Response = self.context.get_raw(url)
        if 'Content-Type' in resp.headers and resp.headers['Content-Type']:
            header_extension: str = '.' + resp.headers['Content-Type'].split(';')[0].split('/')[-1]
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
            comments_list: List[Dict[str, Any]] = sorted(sorted(list(comments), key=lambda t: int(t['id'])),
                                                         key=lambda t: int(t['created_at']), reverse=True)
            unique_comments_list: List[Dict[str, Any]] = [comments_list[0]]
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

        def get_new_comments(new_comments: Iterator[Any], start: int) -> Iterator[Any]:
            for idx, comment in enumerate(new_comments, start=start+1):
                if idx % 250 == 0:
                    self.context.log('{}'.format(idx), end='…', flush=True)
                yield comment

        def save_comments(extended_comments: List[Dict[str, Any]]) -> None:
            unique_comments: List[Dict[str, Any]] = get_unique_comments(extended_comments, combine_answers=True)
            answer_ids: Set[int] = set(int(answer['id']) for comment in unique_comments for answer in comment.get('answers', []))
            with open(filename, 'w') as file:
                file.write(json.dumps(list(filter(lambda t: int(t['id']) not in answer_ids, unique_comments)),
                                      indent=4))

        base_filename: str = filename
        filename += '_comments.json'
        try:
            with open(filename) as fp:
                comments: List[Dict[str, Any]] = json.load(fp)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            comments = list()

        comments_iterator: Iterator[Any] = post.get_comments()
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
            pcaption: str = caption.replace('\n', ' ').strip()
            return '[' + ((pcaption[:29] + "\u2026") if len(pcaption) > 31 else pcaption) + ']'
        filename += '.txt'
        caption += '\n'
        pcaption: str = _elliptify(caption)
        bcaption: bytes = caption.encode("UTF-8")
        with suppress(FileNotFoundError):
            with open(filename, 'rb') as file:
                file_caption: bytes = file.read()
            if file_caption.replace(b'\r\n', b'\n') == bcaption.replace(b'\r\n', b'\n'):
                try:
                    self.context.log(pcaption + ' unchanged', end=' ', flush=True)
                except UnicodeEncodeError:
                    self.context.log('txt unchanged', end=' ', flush=True)
                return
            else:
                def get_filename(index: int) -> str:
                    return filename if index == 0 else '{0}_old_{2:02}{1}'.format(*os.path.splitext(filename), index)
                i: int = 0
                while os.path.isfile(get_filename(i)):
                    i += 1
                for index in range(i, 0, -1):
                    os.rename(get_filename(index - 1), get_filename(index))
                try:
                    self.context.log(_elliptify(file_caption.decode("UTF-8")) + ' updated', end=' ', flush=True)
                except UnicodeEncodeError:
                    self.context.log('txt updated', end=' ', flush=True)
        try:
            self.context.log(pcaption, end=' ', flush=True)
        except UnicodeEncodeError:
            self.context.log('txt', end=' ', flush=True)
        with open(filename, 'w', encoding='UTF-8') as fio:
            fio.write(caption)
        os.utime(filename, (datetime.now().timestamp(), mtime.timestamp()))

    def save_location(self, filename: str, location: PostLocation, mtime: datetime) -> None:
        filename += '_location.txt'
        if location.lat is not None and location.lng is not None:
            location_string: str = (location.name + "\n" +
                               "https://maps.google.com/maps?q={0},{1}&ll={0},{1}\n".format(location.lat,
                                                                                            location.lng))
        else:
            location_string = location.name
        with open(filename, 'wb') as text_file:
            with BytesIO(location_string.encode()) as bio:
                shutil.copyfileobj(cast(IO, bio), text_file)
        os.utime(filename, (datetime.now().timestamp(), mtime.timestamp()))
        self.context.log('geo', end=' ', flush=True)

    def format_filename_within_target_path(self,
                                           target: Union[str, Path],
                                           owner_profile: Optional[Profile],
                                           identifier: str,
                                           name_suffix: str,
                                           extension: str) -> str:
        if ((format_string_contains_key(self.dirname_pattern, 'profile') or
             format_string_contains_key(self.dirname_pattern, 'target'))):
            profile_str: str = owner_profile.username.lower() if owner_profile is not None else str(target)
            return os.path.join(self.dirname_pattern.format(profile=profile_str, target=target),
                                '{0}_{1}.{2}'.format(identifier, name_suffix, extension))
        else:
            return os.path.join(self.dirname_pattern.format(),
                                '{0}_{1}_{2}.{3}'.format(target, identifier, name_suffix, extension))

    @_retry_on_connection_error
    def download_title_pic(self, url: str, target: Union[str, Path], name_suffix: str, owner_profile: Optional[Profile],
                           _attempt: int = 1) -> None:
        http_response: requests.Response = self.context.get_raw(url)
        date_object: Optional[datetime] = None
        if 'Last-Modified' in http_response.headers:
            date_object = datetime.strptime(http_response.headers["Last-Modified"], '%a, %d %b %Y %H:%M:%S GMT')
            date_object = date_object.replace(tzinfo=timezone.utc)
            pic_bytes = None
        else:
            pic_bytes = http_response.content
        ig_filename: str = url.split('/')[-1].split('?')[0]
        pic_data: TitlePic = TitlePic(owner_profile, target, name_suffix, ig_filename, date_object)
        dirname: str = _PostPathFormatter(pic_data, self.sanitize_paths).format(self.dirname_pattern, target=target)
        filename_template: str = os.path.join(
                dirname,
                _PostPathFormatter(pic_data, self.sanitize_paths).format(self.title_pattern, target=target))
        filename: str = self.__prepare_filename(filename_template, lambda: url) + ".jpg"
        content_length: Optional[str] = http_response.headers.get('Content-Length', None)
        if os.path.isfile(filename) and (not self.context.is_logged_in or
                                         (content_length is not None and
                                          os.path.getsize(filename) >= int(content_length))):
            self.context.log(filename + ' already exists')
            return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.context.write_raw(pic_bytes if pic_bytes else http_response, filename)
        if date_object:
            os.utime(filename, (datetime.now().timestamp(), date_object.timestamp()))
        self.context.log('')

    def download_profilepic_if_new(self, profile: Profile, latest_stamps: Optional[LatestStamps]) -> None:
        if latest_stamps is None:
            self.download_profilepic(profile)
            return
        profile_pic_basename: str = profile.profile_pic_url_no_iphone.split('/')[-1].split('?')[0]
        saved_basename: Optional[str] = latest_stamps.get_profile_pic(profile.username)
        if saved_basename == profile_pic_basename:
            return
        self.download_profilepic(profile)
        latest_stamps.set_profile_pic(profile.username, profile_pic_basename)

    def download_profilepic(self, profile: Profile) -> None:
        self.download_title_pic(profile.profile_pic_url, profile.username.lower(), 'profile_pic', profile)

    def download_highlight_cover(self, highlight: Highlight, target: Union[str, Path]) -> None:
        self.download_title_pic(highlight.cover_url, target, 'cover', highlight.owner_profile)

    def download_hashtag_profilepic(self, hashtag: Hashtag) -> None:
        self.download_title_pic(hashtag.profile_pic_url, '#' + hashtag.name, 'profile_pic', None)

    @_requires_login
    def save_session(self) -> dict:
        return self.context.save_session()

    def load_session(self, username: str, session_data: dict) -> None:
        self.context.load_session(username, session_data)

    @_requires_login
    def save_session_to_file(self, filename: Optional[str] = None) -> None:
        if filename is None:
            assert self.context.username is not None
            filename = get_default_session_filename(self.context.username)
        dirname: str = os.path.dirname(filename)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)
            os.chmod(dirname, 0o700)
        with open(filename, 'wb') as sessionfile:
            os.chmod(filename, 0o600)
            self.context.save_session_to_file(sessionfile)
            self.context.log("Saved session to %s." % filename)

    def load_session_from_file(self, username: str, filename: Optional[str] = None) -> None:
        if filename is None:
            filename = get_default_session_filename(username)
            if not os.path.exists(filename):
                filename = get_legacy_session_filename(username)
        with open(filename, 'rb') as sessionfile:
            self.context.load_session_from_file(username, sessionfile)
            self.context.log("Loaded session from %s." % filename)

    def test_login(self) -> Optional[str]:
        return self.context.test_login()

    def login(self, user: str, passwd: str) -> None:
        self.context.login(user, passwd)

    def two_factor_login(self, two_factor_code: str) -> None:
        self.context.two_factor_login(two_factor_code)

    @staticmethod
    def __prepare_filename(filename_template: str, url: Callable[[], str]) -> str:
        if "{filename}" in filename_template:
            filename: str = filename_template.replace("{filename}",
                                                 os.path.splitext(os.path.basename(urlparse(url()).path))[0])
        else:
            filename = filename_template
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return filename

    def format_filename(self, item: Union[Post, StoryItem, PostSidecarNode, TitlePic],
                        target: Optional[Union[str, Path]] = None) -> str:
        return _PostPathFormatter(item, self.sanitize_paths).format(self.filename_pattern, target=target)

    def download_post(self, post: Post, target: Union[str, Path]) -> bool:
        def _already_downloaded(path: str) -> bool:
            if not os.path.isfile(path):
                return False
            else:
                self.context.log(path + ' exists', end=' ', flush=True)
                return True

        def _all_already_downloaded(path_base: str, is_videos_enumerated: Iterator[Any]) -> bool:
            if '{filename}' in self.filename_pattern:
                return False
            for idx, is_video in is_videos_enumerated:
                if self.download_pictures and (not is_video or self.download_video_thumbnails):
                    if not _already_downloaded("{0}_{1}.jpg".format(path_base, idx)):
                        return False
                if is_video and self.download_videos:
                    if not _already_downloaded("{0}_{1}.mp4".format(path_base, idx)):
                        return False
            return True

        dirname: str = _PostPathFormatter(post, self.sanitize_paths).format(self.dirname_pattern, target=target)
        filename_template: str = os.path.join(dirname, self.format_filename(post, target=target))
        filename: str = self.__prepare_filename(filename_template, lambda: post.url)
        downloaded: bool = True
        if post.typename == 'GraphSidecar':
            if (self.download_pictures or self.download_videos) and post.mediacount > 0:
                if not _all_already_downloaded(
                        filename_template, enumerate(
                            (post.get_is_videos()[i]
                             for i in range(self.slide_start % post.mediacount, self.slide_end % post.mediacount + 1)),
                            start=self.slide_start % post.mediacount + 1
                        )
                ):
                    for edge_number, sidecar_node in enumerate(
                            post.get_sidecar_nodes(self.slide_start, self.slide_end),
                            start=self.slide_start % post.mediacount + 1
                    ):
                        suffix: Optional[str] = str(edge_number)
                        if '{filename}' in self.filename_pattern:
                            suffix = None
                        if self.download_pictures and (not sidecar_node.is_video or self.download_video_thumbnails):
                            sidecar_filename: str = self.__prepare_filename(filename_template,
                                                                           lambda: sidecar_node.display_url)
                            downloaded &= self.download_pic(filename=sidecar_filename, url=sidecar_node.display_url,
                                                            mtime=post.date_local, filename_suffix=suffix)
                        if sidecar_node.is_video and self.download_videos:
                            sidecar_filename = self.__prepare_filename(filename_template,
                                                                       lambda: sidecar_node.video_url)
                            downloaded &= self.download_pic(filename=sidecar_filename, url=sidecar_node.video_url,
                                                            mtime=post.date_local, filename_suffix=suffix)
                else:
                    downloaded = False
        elif post.typename == 'GraphImage':
            if self.download_pictures:
                downloaded = (not _already_downloaded(filename + ".jpg") and
                              self.download_pic(filename=filename, url=post.url, mtime=post.date_local))
        elif post.typename == 'GraphVideo':
            if self.download_pictures and self.download_video_thumbnails:
                with self.context.error_catcher("Video thumbnail of {}".format(post)):
                    downloaded = (not _already_downloaded(filename + ".jpg") and
                                  self.download_pic(filename=filename, url=post.url, mtime=post.date_local))
        else:
            self.context.error("Warning: {0} has unknown typename: {1}".format(post, post.typename))

        metadata_string: str = _ArbitraryItemFormatter(post).format(self.post_metadata_txt_pattern).strip()
        if metadata_string:
            self.save_caption(filename=filename, mtime=post.date_local, caption=metadata_string)

        if post.is_video and self.download_videos:
            downloaded &= (not _already_downloaded(filename + ".mp4") and
                           self.download_pic(filename=filename, url=post.video_url, mtime=post.date_local))

        if self.download_geotags and post.location:
            self.save_location(filename, post.location, post.date_local)

        if self.download_comments:
            self.update_comments(filename=filename, post=post)

        if self.save_metadata:
            self.save_metadata_json(filename, post)

        self.context.log()
        return downloaded

    @_requires_login
    def get_stories(self, userids: Optional[List[int]] = None) -> Iterator[Story]:
        if not userids:
            data: dict = self.context.graphql_query("d15efd8c0c5b23f0ef71f18bf363c704",
                                              {"only_stories": True})["data"]["user"]
            if data is None:
                raise BadResponseException('Bad stories reel JSON.')
            userids = list(edge["node"]["id"] for edge in data["feed_reels_tray"]["edge_reels_tray_to_reel"]["edges"])

        def _userid_chunks() -> Iterator[List[int]]:
            assert userids is not None
            userids_per_query: int = 50
            for i in range(0, len(userids), userids_per_query):
                yield userids[i:i + userids_per_query]

        for userid_chunk in _userid_chunks():
            stories: dict = self.context.graphql_query("303a4ae99711322310f25250d988f3b7",
                                                 {"reel_ids": userid_chunk, "precomposed_overlay": False})["data"]
            yield from (Story(self.context, media) for media in stories['reels_media'])

    @_requires_login
    def download_stories(self,
                         userids: Optional[List[Union[int, Profile]]] = None,
                         fast_update: bool = False,
                         filename_target: Optional[str] = ':stories',
                         storyitem_filter: Optional[Callable[[StoryItem], bool]] = None,
                         latest_stamps: Optional[LatestStamps] = None) -> None:
        if not userids:
            self.context.log("Retrieving all visible stories...")
            profile_count: Optional[int] = None
        else:
            userids = [p if isinstance(p, int) else p.userid for p in userids]
            profile_count = len(userids)

        for i, user_story in enumerate(self.get_stories(userids), start=1):
            name: str = user_story.owner_username
            if profile_count is not None:
                msg: str = "[{0:{w}d}/{1:{w}d}] Retrieving stories from profile {2}.".format(i, profile_count, name,
                                                                                        w=len(str(profile_count)))
            else:
                msg = "[{:3d}] Retrieving stories from profile {}.".format(i, name)
            self.context.log(msg)
            totalcount: int = user_story.itemcount
            count: int = 1
            if latest_stamps is not None:
                last_scraped: datetime = latest_stamps.get_last_story_timestamp(name)
                scraped_timestamp: datetime = datetime.now().astimezone()
            for item in user_story.get_items():
                if latest_stamps is not None:
                    if item.date_local <= last_scraped:
                        break
                if storyitem_filter is not None and not storyitem_filter(item):
                    self.context.log("<{} skipped>".format(item), flush=True)
                    continue
                self.context.log("[%3i/%3i] " % (count, totalcount), end="", flush=True)
                count += 1
                with self.context.error_catcher('Download story from user {}'.format(name)):
                    downloaded: bool = self.download_storyitem(item, filename_target if filename_target else name)
                    if fast_update and not downloaded:
                        break
            if latest_stamps is not None:
                latest_stamps.set_last_story_timestamp(name, scraped_timestamp)

    def download_storyitem(self, item: StoryItem, target: Union[str, Path]) -> bool:
        def _already_downloaded(path: str) -> bool:
            if not os.path.isfile(path):
                return False
            else:
                self.context.log(path + ' exists', end=' ', flush=True)
                return True

        date_local: datetime = item.date_local
        dirname: str = _PostPathFormatter(item, self.sanitize_paths).format(self.dirname_pattern, target=target)
        filename_template: str = os.path.join(dirname, self.format_filename(item, target=target))
        filename: str = self.__prepare_filename(filename_template, lambda: item.url)
        downloaded: bool = False
        video_url_fetch_failed: bool = False
        if item.is_video and self.download_videos is True:
            video_url: Optional[str] = item.video_url
            if video_url:
                filename = self.__prepare_filename(filename_template, lambda: str(video_url))
                downloaded |= (not _already_downloaded(filename + ".mp4") and
                               self.download_pic(filename=filename, url=video_url, mtime=date_local))
            else:
                video_url_fetch_failed = True
        if video_url_fetch_failed or not item.is_video or self.download_video_thumbnails is True:
            downloaded = (not _already_downloaded(filename + ".jpg") and
                          self.download_pic(filename=filename, url=item.url, mtime=date_local))
        metadata_string: str = _ArbitraryItemFormatter(item).format(self.storyitem_metadata_txt_pattern).strip()
        if metadata_string:
            self.save_caption(filename=filename, mtime=item.date_local, caption=metadata_string)
        if self.save_metadata is not False:
            self.save_metadata_json(filename, item)
        self.context.log()
        return downloaded

    @_requires_login
    def get_highlights(self, user: Union[int, Profile]) -> Iterator[Highlight]:
        userid: int = user if isinstance(user, int) else user.userid
        data: dict = self.context.graphql_query("7c16654f22c819fb63d1183034a5162f",
                                          {"user_id": userid, "include_chaining": False, "include_reel": False,
                                           "include_suggested_users": False, "include_logged_out_extras": False,
                                           "include_highlight_reels": True})["data"]["user"]['edge_highlight_reels']
        if data is None:
            raise BadResponseException('Bad highlights reel JSON.')
        yield from (Highlight(self.context, edge['node'], user if isinstance(user, Profile) else None)
                    for edge in data['edges'])

    @_requires_login
    def download_highlights(self,
                            user: Union[int, Profile],
                            fast_update: bool = False,
                            filename_target: Optional[str] = None,
                            storyitem_filter: Optional[Callable[[StoryItem], bool]] = None) -> None:
        for user_highlight in self.get_highlights(user):
            name: str = user_highlight.owner_username
            highlight_target: Union[str, Path] = (filename_target
                                if filename_target
                                else (Path(_PostPathFormatter.sanitize_path(name, self.sanitize_paths)) /
                                      _PostPathFormatter.sanitize_path(user_highlight.title, self.sanitize_paths)))
            self.context.log("Retrieving highlights \"{}\" from profile {}".format(user_highlight.title, name))
            self.download_highlight_cover(user_highlight, highlight_target)
            totalcount: int = user_highlight.itemcount
            count: int = 1
            for item in user_highlight.get_items():
                if storyitem_filter is not None and not storyitem_filter(item):
                    self.context.log("<{} skipped>".format(item), flush=True)
                    continue
                self.context.log("[%3i/%3i] " % (count, totalcount), end="", flush=True)
                count += 1
                with self.context.error_catcher('Download highlights \"{}\" from user {}'.format(user_highlight.title,
                                                                                                 name)):
                    downloaded: bool = self.download_storyitem(item, highlight_target)
                    if fast_update and not downloaded:
                        break

    def posts_download_loop(self,
                            posts: Iterator[Post],
                            target: Union[str, Path],
                            fast_update: bool = False,
                            post_filter: Optional[Callable[[Post], bool]] = None,
                            max_count: Optional[int] = None,
                            total_count: Optional[int] = None,
                            owner_profile: Optional[Profile] = None,
                            takewhile: Optional[Callable[[Post], bool]] = None,
                            possibly_pinned: int = 0) -> None:
        displayed_count: Optional[int] = (max_count if total_count is None or (max_count is not None and max_count < total_count)
                           else total_count)
        sanitized_target: Union[str, Path] = target
        if isinstance(target, str):
            sanitized_target = _PostPathFormatter.sanitize_path(target, self.sanitize_paths)
        if takewhile is None:
            takewhile = lambda _: True
        with resumable_iteration(
                context=self.context,
                iterator=posts,
                load=load_structure_from_file,
                save=save_structure_to_file,
                format_path=lambda magic: self.format_filename_within_target_path(
                    sanitized_target, owner_profile, self.resume_prefix or '', magic, 'json.xz'
                ),
                check_bbd=self.check_resume_bbd,
                enabled=self.resume_prefix is not None
        ) as (is_resuming, start_index):
            for number, post in enumerate(posts, start=start_index + 1):
                should_stop: bool = not takewhile(post)
                if should_stop and number <= possibly_pinned:
                    continue
                if (max_count is not None and number > max_count) or should_stop:
                    break
                if displayed_count is not None:
                    self.context.log("[{0:{w}d}/{1:{w}d}] ".format(number, displayed_count,
                                                                   w=len(str(displayed_count))),
                                     end="", flush=True)
                else:
                    self.context.log("[{:3d}] ".format(number), end="", flush=True)
                if post_filter is not None:
                    try:
                        if not post_filter(post):
                            self.context.log("{} skipped".format(post))
                            continue
                    except (InstaloaderException, KeyError, TypeError) as err:
                        self.context.error("{} skipped. Filter evaluation failed: {}".format(post, err))
                        continue
                with self.context.error_catcher("Download {} of {}".format(post, target)):
                    post_changed: bool = False
                    while True:
                        try:
                            downloaded: bool = self.download_post(post, target=target)
                            break
                        except PostChangedException:
                            post_changed = True
                            continue
                    if fast_update and not downloaded and not post_changed and number > possibly_pinned:
                        if not is_resuming or number > 0:
                            break

    @_requires_login
    def get_feed_posts(self) -> Iterator[Post]:
        data: dict = self.context.graphql_query("d6f4427fbe92d846298cf93df0b937d3", {})["data"]
        while True:
            feed: dict = data["user"]["edge_web_feed_timeline"]
            for edge in feed["edges"]:
                node: dict = edge["node"]
                if node.get("__typename") in Post.supported_graphql_types() and node.get("shortcode") is not None:
                    yield Post(self.context, node)
            if not feed["page_info"]["has_next_page"]:
                break
            data = self.context.graphql_query("d6f4427fbe92d846298cf93df0b937d3",
                                              {'fetch_media_item_count': 12,
                                               'fetch_media_item_cursor': feed["page_info"]["end_cursor"],
                                               'fetch_comment_count': 4,
                                               'fetch_like': 10,
                                               'has_stories': False})["data"]

    @_requires_login
    def download_feed_posts(self, max_count: Optional[int] = None, fast_update: bool = False,
                            post_filter: Optional[Callable[[Post], bool]] = None) -> None:
        self.context.log("Retrieving pictures from your feed...")
        self.posts_download_loop(self.get_feed_posts(), ":feed", fast_update, post_filter, max_count=max_count)

    @_requires_login
    def download_saved_posts(self, max_count: Optional[int] = None, fast_update: bool = False,
                             post_filter: Optional[Callable[[Post], bool]] = None) -> None:
        self.context.log("Retrieving saved posts...")
        assert self.context.username is not None
        node_iterator: NodeIterator[Post] = Profile.own_profile(self.context).get_saved_posts()
        self.posts_download_loop(node_iterator, ":saved",
                                 fast_update, post_filter,
                                 max_count=max_count, total_count=node_iterator.count)

    @_requires_login
    def get_location_posts(self, location: str) -> Iterator[Post]:
        yield from SectionIterator(
            self.context,
            lambda d: d["native_location_data"]["recent"],
            lambda m: Post.from_iphone_struct(self.context, m),
            f"explore/locations/{location}/",
        )

    @_requires_login
    def download_location(self, location: str,
                          max_count: Optional[int] = None,
                          post_filter: Optional[Callable[[Post], bool]] = None,
                          fast_update: bool = False) -> None:
        self.context.log("Retrieving pictures for location {}...".format(location))
        self.posts_download_loop(self.get_location_posts(location), "%" + location, fast_update, post_filter,
                                 max_count=max_count)

    @_requires_login
    def get_explore_posts(self) -> NodeIterator[Post]:
        return NodeIterator(
            self.context,
            'df0dcc250c2b18d9fd27c5581ef33c7c',
            lambda d: d['data']['user']['edge_web_discover_media'],
            lambda n: Post(self.context, n),
            query_referer='https://www.instagram.com/explore/',
        )

    def get_hashtag_posts(self, hashtag: str) -> Iterator[Post]:
        return Hashtag.from_name(self.context, hashtag).get_posts_resumable()

    @_requires_login
    def download_hashtag(self, hashtag: Union[Hashtag, str],
                         max_count: Optional[int] = None,
                         post_filter: Optional[Callable[[Post], bool]] = None,
                         fast_update: bool = False,
                         profile_pic: bool = True,
                         posts: bool = True) -> None:
        if isinstance(hashtag, str):
            with self.context.error_catcher("Get hashtag #{}".format(hashtag)):
                hashtag = Hashtag.from_name(self.context, hashtag)
        if not isinstance(hashtag, Hashtag):
            return
        target: str = "#" + hashtag.name
        if profile_pic:
            with self.context.error_catcher("Download profile picture of {}".format(target)):
                self.download_hashtag_profilepic(hashtag)
        if posts:
            self.context.log("Retrieving pictures with hashtag #{}...".format(hashtag.name))
            self.posts_download_loop(hashtag.get_posts_resumable(), target, fast_update, post_filter,
                                     max_count=max_count)
        if self.save_metadata:
            json_filename: str = '{0}/{1}'.format(self.dirname_pattern.format(profile=target,
                                                                         target=target),
                                             target)
            self.save_metadata_json(json_filename, hashtag)

    def download_tagged(self, profile: Profile, fast_update: bool = False,
                        target: Optional[str] = None,
                        post_filter: Optional[Callable[[Post], bool]] = None,
                        latest_stamps: Optional[LatestStamps] = None) -> None:
        self.context.log("Retrieving tagged posts for profile {}.".format(profile.username))
        posts_takewhile: Optional[Callable[[Post], bool]] = None
        if latest_stamps is not None:
            last_scraped: datetime = latest_stamps.get_last_tagged_timestamp(profile.username)
            posts_takewhile = lambda p: p.date_local > last_scraped
        tagged_posts: Iterator[Post] = profile.get_tagged_posts()
        self.posts_download_loop(tagged_posts,
                                 target if target
                                 else (Path(_PostPathFormatter.sanitize_path(profile.username, self.sanitize_paths)) /
                                       _PostPathFormatter.sanitize_path(':tagged', self.sanitize_paths)),
                                 fast_update, post_filter, takewhile=posts_takewhile)
        if latest_stamps is not None and tagged_posts.first_item is not None:
            latest_stamps.set_last_tagged_timestamp(profile.username, tagged_posts.first_item.date_local)

    def download_reels(self, profile: Profile, fast_update: bool = False,
                      post_filter: Optional[Callable[[Post], bool]] = None,
                      latest_stamps: Optional[LatestStamps] = None) -> None:
        self.context.log("Retrieving reels videos for profile {}.".format(profile.username))
        posts_takewhile: Optional[Callable[[Post], bool]] = None
        if latest_stamps is not None:
            last_scraped: datetime = latest_stamps.get_last_reels_timestamp(profile.username)
            posts_takewhile = lambda p: p.date_local > last_scraped
        reels: Iterator[Post] = profile.get_reels()
        self.posts_download_loop(
            reels,
            profile.username,
            fast_update,
            post_filter,
            owner_profile=profile,
            takewhile=posts_takewhile,
        )
        if latest_stamps is not None and reels.first_item is not None:
            latest_stamps.set_last_reels_timestamp(profile.username, reels.first_item.date_local)

    def download_igtv(self, profile: Profile, fast_update: bool = False,
                      post_filter: Optional[Callable[[Post], bool]] = None,
                      latest_stamps: Optional[LatestStamps] = None) -> None:
        self.context.log("Retrieving IGTV videos for profile {}.".format(profile.username))
        posts_takewhile: Optional[Callable[[Post], bool]] = None
        if latest_stamps is not None:
            last_scraped: datetime = latest_stamps.get_last_igtv_timestamp(profile.username)
            posts_takewhile = lambda p: p.date_local > last_scraped
        igtv_posts: Iterator[Post] = profile.get_igtv_posts()
        self.posts_download_loop(igtv_posts, profile.username, fast_update, post_filter,
                                 total_count=profile.igtvcount, owner_profile=profile, takewhile=posts_takewhile)
        if latest_stamps is not None and igtv_posts.first_item is not None:
            latest_stamps.set_last_igtv_timestamp(profile.username, igtv_posts.first_item.date_local)

    def _get_id_filename(self, profile_name: str) -> str:
        if ((format_string_contains_key(self.dirname_pattern, 'profile') or
             format_string_contains_key(self.dirname_pattern, 'target'))):
            return os.path.join(self.dirname_pattern.format(profile=profile_name.lower(),
                                                            target=profile_name.lower()),
                                'id')
        else:
            return os.path.join(self.dirname_pattern.format(),
                                '{0}_id'.format(profile_name.lower()))

    def load_profile_id(self, profile_name: str) -> Optional[int]:
        id_filename: str = self._get_id_filename(profile_name)
        try:
            with open(id_filename, 'rb') as id_file:
                return int(id_file.read())
        except (FileNotFoundError, ValueError):
            return None

    def save_profile_id(self, profile: Profile) -> None:
        os.makedirs(self.dirname_pattern.format(profile=profile.username,
                                                target=profile.username), exist_ok=True)
        with open(self._get_id_filename(profile.username), 'w') as text_file:
            text_file.write(str(profile.userid) + "\n")
            self.context.log("Stored ID {0} for profile {1}.".format(profile.userid, profile.username))

    def check_profile_id(self, profile_name: str, latest_stamps: Optional[LatestStamps] = None) -> Profile:
        profile: Optional[Profile] = None
        profile_name_not_exists_err: Optional[Exception] = None
        try:
            profile = Profile.from_username(self.context, profile_name)
        except ProfileNotExistsException as err:
            profile_name_not_exists_err = err
        if latest_stamps is None:
            profile_id: Optional[int] = self.load_profile_id(profile_name)
        else:
            profile_id = latest_stamps.get_profile_id(profile_name)
        if profile_id is not None:
            if (profile is None) or (profile_id != profile.userid):
                if profile is not None:
                    self.context.log("Profile {0} does not match the stored unique ID {1}.".format(profile_name,
                                                                                                   profile_id))
                else:
                    self.context.log("Trying to find profile {0} using its unique ID {1}.".format(profile_name,
                                                                                                  profile_id))
                profile_from_id: Profile = Profile.from_id(self.context, profile_id)
                newname: str = profile_from_id.username
                if profile_name == newname:
                    self.context.error(
                        f"Warning: Profile {profile_name} could not be retrieved by its name, but by its ID.")
                    return profile_from_id
                self.context.error("Profile {0} has changed its name to {1}.".format(profile_name, newname))
                if latest_stamps is None:
                    if ((format_string_contains_key(self.dirname_pattern, 'profile') or
                         format_string_contains_key(self.dirname_pattern, 'target'))):
                        os.rename(self.dirname_pattern.format(profile=profile_name.lower(),
                                                              target=profile_name.lower()),
                                  self.dirname_pattern.format(profile=newname.lower(),
                                                              target=newname.lower()))
                    else:
                        os.rename('{0}/{1}_id'.format(self.dirname_pattern.format(), profile_name.lower()),
                                  '{0}/{1}_id'.format(self.dirname_pattern.format(), newname.lower()))
                else:
                    latest_stamps.rename_profile(profile_name, newname)
                return profile_from_id
            return profile
        if profile is not None:
            if latest_stamps is None:
                self.save_profile_id(profile)
            else:
                latest_stamps.save_profile_id(profile.username, profile.userid)
            return profile
        if profile_name_not_exists_err:
            raise profile_name_not_exists_err
        raise ProfileNotExistsException("Profile {0} does not exist.".format(profile_name))

    def download_profiles(self, profiles: Set[Profile],
                          profile_pic: bool = True, posts: bool = True,
                          tagged: bool = False,
                          igtv: bool = False,
                          highlights: bool = False,
                          stories: bool = False,
                          fast_update: bool = False,
                          post_filter: Optional[Callable[[Post], bool]] = None,
                          storyitem_filter: Optional[Callable[[StoryItem], bool]] = None,
                          raise_errors: bool = False,
                          latest_stamps: Optional[LatestStamps] = None,
                          max_count: Optional[int] = None,
                          reels: bool = False) -> None:

        @contextmanager
        def _error_raiser(_str: Optional[str]) -> Iterator[None]:
            yield

        error_handler: Callable[[Optional[str]], ContextManager[None]] = _error_raiser if raise_errors else self.context.error_catcher

        for i, profile in enumerate(profiles, start=1):
            self.context.log("[{0:{w}d}/{1:{w}d}] Downloading profile {2}".format(i, len(profiles), profile.username,
                                                                                  w=len(str(len(profiles)))))
            with error_handler(profile.username):
                profile_name: str = profile.username
                if profile_pic:
                    with self.context.error_catcher('Download profile picture of {}'.format(profile_name)):
                        self.download_profilepic_if_new(profile, latest_stamps)
                if self.save_metadata:
                    json_filename: str = os.path.join(self.dirname_pattern.format(profile=profile_name,
                                                                             target=profile_name),
                                                 '{0}_{1}'.format(profile_name, profile.userid))
                    self.save_metadata_json(json_filename, profile)
                if tagged or igtv or highlights or posts:
                    if (not self.context.is_logged_in and profile.is_private):
                        raise LoginRequiredException("Login required.")
                    if (self.context.username != profile.username and
                            profile.is_private and
                            not profile.followed_by_viewer):
                        raise PrivateProfileNotFollowedException("Private but not followed.")
                if tagged:
                    with self.context.error_catcher('Download tagged of {}'.format(profile_name)):
                        self.download_tagged(profile, fast_update=fast_update, post_filter=post_filter,
                                             latest_stamps=latest_stamps)
                if reels:
                    with self.context.error_catcher('Download reels of {}'.format(profile_name)):
                        self.download_reels(profile, fast_update=fast_update, post_filter=post_filter,
                                           latest_stamps=latest_stamps)
                if igtv:
                    with self.context.error_catcher('Download IGTV of {}'.format(profile_name)):
                        self.download_igtv(profile, fast_update=fast_update, post_filter=post_filter,
                                           latest_stamps=latest_stamps)
                if highlights:
                    with self.context.error_catcher('Download highlights of {}'.format(profile_name)):
                        self.download_highlights(profile, fast_update=fast_update, storyitem_filter=storyitem_filter)
                if posts:
                    self.context.log("Retrieving posts from profile {}.".format(profile_name))
                    posts_takewhile: Optional[Callable[[Post], bool]] = None
                    if latest_stamps is not None:
                        last_scraped: datetime = latest_stamps.get_last_post_timestamp(profile_name)
                        posts_takewhile = lambda p: p.date_local > last_scraped
                    posts_to_download: Iterator[Post] = profile.get_posts()
                    self.posts_download_loop(posts_to_download, profile_name, fast_update, post_filter,
                                             total_count=profile.mediacount, owner_profile=profile,
                                             takewhile=posts_takewhile, possibly_pinned=3, max_count=max_count)
                    if latest_stamps is not None and posts_to_download.first_item is not None:
                        latest_stamps.set_last_post_timestamp(profile_name,
                                                              posts_to_download.first_item.date_local)

        if stories and profiles:
            with self.context.error_catcher("Download stories"):
                self.context.log("Downloading stories")
                self.download_stories(userids=list(profiles), fast_update=fast_update, filename_target=None,
                                      storyitem_filter=storyitem_filter, latest_stamps=latest_stamps)

    def download_profile(self, profile_name: Union[str, Profile],
                         profile_pic: bool = True, profile_pic_only: bool = False,
                         fast_update: bool = False,
                         download_stories: bool = False, download_stories_only: bool = False,
                         download_tagged: bool = False, download_tagged_only: bool = False,
                         post_filter: Optional[Callable[[Post], bool]] = None,
                         storyitem_filter: Optional[Callable[[StoryItem], bool]] = None) -> None:
        if isinstance(profile_name, str):
            profile: Profile = self.check_profile_id(profile_name.lower())
        else:
            profile = profile_name
        profile_name = profile.username
        if self.save_metadata is not False:
            json_filename: str = '{0}/{1}_{2}'.format(self.dirname_pattern.format(profile=profile_name, target=profile_name),
                                                 profile_name, profile.userid)
            self.save_metadata_json(json_filename, profile)
        if self.context.is_logged_in and profile.has_blocked_viewer and not profile.is_private:
            raise ProfileNotExistsException("Profile {} has blocked you".format(profile_name))
        if profile_pic or profile_pic_only:
            with self.context.error_catcher('Download profile picture of {}'.format(profile_name)):
                self.download_profilepic(profile)
        if profile_pic_only:
            return
        if profile.is_private:
            if not self.context.is_logged_in:
                raise LoginRequiredException("profile %s requires login" % profile_name)
            if not profile.followed_by_viewer and \
                    self.context.username != profile.username:
                raise PrivateProfileNotFollowedException("Profile %s: private but not followed." % profile_name)
        else:
            if self.context.is_logged_in and not (download_stories or download_stories_only):
                self.context.log("profile %s could also be downloaded anonymously." % profile_name)
        if download_stories or download_stories_only:
            if profile.has_viewable_story:
                with self.context.error_catcher("Download stories of {}".format(profile_name)):
                    self.download_stories(userids=[profile.userid], filename_target=profile_name,
                                          fast_update=fast_update, storyitem_filter=storyitem_filter)
            else:
                self.context.log("{} does not have any stories.".format(profile_name))
        if download_stories_only:
            return
        if download_tagged or download_tagged_only:
            with self.context.error_catcher('Download tagged of {}'.format(profile_name)):
                self.download_tagged(profile, fast_update=fast_update, post_filter=post_filter)
        if download_tagged_only:
            return
        self.context.log("Retrieving posts from profile {}.".format(profile_name))
        self.posts_download_loop(profile.get_posts(), profile_name, fast_update, post_filter,
                                 total_count=profile.mediacount, owner_profile=profile)

    def interactive_login(self, username: str) -> None:
        if self.context.quiet:
            raise InvalidArgumentException("Quiet mode requires given password or valid session file.")
        try:
            password: Optional[str] = None
            while password is None:
                password = getpass.getpass(prompt="Enter Instagram password for %s: " % username)
                try:
                    self.login(username, password)
                except BadCredentialsException as err:
                    print(err, file=sys.stderr)
                    password = None
        except TwoFactorAuthRequiredException:
            while True:
                try:
                    code: str = input("Enter 2FA verification code: ")
                    self.two_factor_login(code)
                    break
                except BadCredentialsException as err:
                    print(err, file=sys.stderr)
                    pass

    @property
    def has_stored_errors(self) -> bool:
        return self.context.has_stored_errors
