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
from typing import Any, Callable, IO, Iterator, List, Optional, Set, Union, cast
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
    configdir = _get_config_dir()
    sessionfilename = "session-{}".format(username)
    return os.path.join(configdir, sessionfilename)


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
        new_loader.context.error_log = []  # avoid double-printing of errors
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
        def _postcommentanswer_asdict(comment: Any) -> dict:
            return {'id': comment.id,
                    'created_at': int(comment.created_at_utc.replace(tzinfo=timezone.utc).timestamp()),
                    'text': comment.text,
                    'owner': comment.owner._asdict(),
                    'likes_count': comment.likes_count}

        def _postcomment_asdict(comment: Any) -> dict:
            return {**_postcommentanswer_asdict(comment),
                    'answers': sorted([_postcommentanswer_asdict(answer) for answer in comment.answers],
                                      key=lambda t: int(t['id']),
                                      reverse=True)}

        def get_unique_comments(comments: List[dict], combine_answers: bool = False) -> List[dict]:
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

        def get_new_comments(new_comments: Iterator[Any], start: int) -> Iterator[dict]:
            for idx, comment in enumerate(new_comments, start=start+1):
                if idx % 250 == 0:
                    self.context.log('{}'.format(idx), end='…', flush=True)
                yield comment

        def save_comments(extended_comments: List[dict]) -> None:
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
        bcaption = caption.encode("UTF-8")
        with suppress(FileNotFoundError):
            with open(filename, 'rb') as file:
                file_caption = file.read()
            if file_caption.replace(b'\r\n', b'\n') == bcaption.replace(b'\r\n', b'\n'):
                try:
                    self.context.log(pcaption + ' unchanged', end=' ', flush=True)
                except UnicodeEncodeError:
                    self.context.log('txt unchanged', end=' ', flush=True)
                return None
            else:
                def get_filename(index: int) -> str:
                    return filename if index == 0 else '{0}_old_{2:02}{1}'.format(*os.path.splitext(filename), index)

                i = 0
                while os.path.isfile(get_filename(i)):
                    i = i + 1
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
            location_string = (location.name + "\n" +
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
            profile_str = owner_profile.username.lower() if owner_profile is not None else target
            return os.path.join(self.dirname_pattern.format(profile=profile_str, target=target),
                                '{0}_{1}.{2}'.format(identifier, name_suffix, extension))
        else:
            return os.path.join(self.dirname_pattern.format(),
                                '{0}_{1}_{2}.{3}'.format(target, identifier, name_suffix, extension))

    @_retry_on_connection_error
    def download_title_pic(self, url: str, target: Union[str, Path], name_suffix: str, owner_profile: Optional[Profile],
                           _attempt: int = 1) -> None:
        http_response = self.context.get_raw(url)
        date_object: Optional[datetime] = None
        if 'Last-Modified' in http_response.headers:
            date_object = datetime.strptime(http_response.headers["Last-Modified"], '%a, %d %b %Y %H:%M:%S GMT')
            date_object = date_object.replace(tzinfo=timezone.utc)
            pic_bytes = None
        else:
            pic_bytes = http_response.content
        ig_filename = url.split('/')[-1].split('?')[0]
        pic_data = TitlePic(owner_profile, target, name_suffix, ig_filename, date_object)
        dirname = _PostPathFormatter(pic_data, self.sanitize_paths).format(self.dirname_pattern, target=target)
        filename_template = os.path.join(
                dirname,
                _PostPathFormatter(pic_data, self.sanitize_paths).format(self.title_pattern, target=target))
        filename = self.__prepare_filename(filename_template, lambda: url) + ".jpg"
        content_length = http_response.headers.get('Content-Length', None)
        if os.path.isfile(filename) and (not self.context.is_logged_in or
                                         (content_length is not None and
                                          os.path.getsize(filename) >= int(content_length))):
            self.context.log(filename + ' already exists')
            return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.context.write_raw(pic_bytes if pic_bytes else http_response, filename)
        if date_object:
            os.utime(filename, (datetime.now().timestamp(), date_object.timestamp()))
        self.context.log('')  # log output of _get_and_write_raw() does not produce \n

    def download_profilepic_if_new(self, profile: Profile, latest_stamps: Optional[LatestStamps]) -> None:
        if latest_stamps is None:
            self.download_profilepic(profile)
            return
        profile_pic_basename = profile.profile_pic_url_no_iphone.split('/')[-1].split('?')[0]
        saved_basename = latest_stamps.get_profile_pic(profile.username)
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
        dirname = os.path.dirname(filename)
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
            filename = filename_template.replace("{filename}",
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

        def _all_already_downloaded(path_base: str, is_videos_enumerated: Iterator[Tuple[int, bool]]) -> bool:
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

        dirname = _PostPathFormatter(post, self.sanitize_paths).format(self.dirname_pattern, target=target)
        filename_template = os.path.join(dirname, self.format_filename(post, target=target))
        filename = self.__prepare_filename(filename_template, lambda: post.url)

        downloaded = True
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
                            sidecar_filename = self.__prepare_filename(filename_template,
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

        metadata_string = _ArbitraryItemFormatter(post).format(self.post_metadata_txt_pattern).strip()
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
            data = self.context.graphql_query("d15efd8c0c5b23f0ef71f18bf363c704",
                                              {"only_stories": True})["data"]["user"]
            if data is None:
                raise BadResponseException('Bad stories reel JSON.')
            userids = list(edge["node"]["id"] for edge in data["feed_reels_tray"]["edge_reels_tray_to_reel"]["edges"])

        def _userid_chunks() -> Iterator[List[int]]:
            assert userids is not None
            userids_per_query = 50
            for i in range(0, len(userids), userids_per_query):
                yield userids[i:i + userids_per_query]

        for userid_chunk in _userid_chunks():
            stories = self.context.graphql_query("303a4ae99711322310f25250d988f3b7",
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
            profile_count = None
        else:
            userids = [p if isinstance(p, int) else p.userid for p in userids]
            profile_count = len(userids)

        for i, user_story in enumerate(self.get_stories(userids), start=1):
            name = user_story.owner_username
            if profile_count is not None:
                msg = "[{0:{w}d}/{1:{w}d}] Retrieving stories from profile {2}.".format(i, profile_count, name,
                                                                                        w=len(str(profile_count)))
            else:
                msg = "[{:3d}] Retrieving stories from profile {}.".format(i, name)
            self.context.log(msg)
            totalcount = user_story.itemcount
            count = 1
            if latest_stamps is not None:
                last_scraped = latest_stamps.get_last_story_timestamp(name)
                scraped_timestamp = datetime.now().astimezone()
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
                    downloaded = self.download_storyitem(item, filename_target if filename_target else name)
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

        date_local = item.date_local
        dirname = _PostPathFormatter(item, self.sanitize_paths).format(self.dirname_pattern, target=target)
        filename_template = os.path.join(dirname, self.format_filename(item, target=target))
        filename = self.__prepare_filename(filename_template, lambda: item.url)
        downloaded = False
        video_url_fetch_failed = False
        if item.is_video and self.download_videos is True:
            video_url = item.video_url
            if video_url:
                filename = self.__prepare_filename(filename_template, lambda: str(video_url))
                downloaded |= (not _already_downloaded(filename + ".mp4") and
                               self.download_pic(filename=filename, url=video_url, mtime=date_local))
            else:
                video_url_fetch_failed = True
        if video_url_fetch_f