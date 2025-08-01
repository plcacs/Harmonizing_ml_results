#!/usr/bin/env python3
"""
网易云音乐 Player
"""
import os
import random
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import logger
from .api import NetEase
from .cache import Cache
from .config import Config
from .kill_thread import stop_thread
from .storage import Storage
from .ui import Ui
from .utils import notify

log = logger.getLogger(__name__)


class Player(object):
    MODE_ORDERED: int = 0
    MODE_ORDERED_LOOP: int = 1
    MODE_SINGLE_LOOP: int = 2
    MODE_RANDOM: int = 3
    MODE_RANDOM_LOOP: int = 4
    SUBPROCESS_LIST: List[Any] = []
    MUSIC_THREADS: List[threading.Thread] = []

    def __init__(self) -> None:
        self.config: Config = Config()
        self.ui: Ui = Ui()
        self.popen_handler: Optional[subprocess.Popen] = None
        self.playing_flag: bool = False
        self.refresh_url_flag: bool = False
        self.process_length: float = 0
        self.process_location: float = 0
        self.storage: Storage = Storage()
        self.cache: Cache = Cache()
        self.end_callback: Optional[Callable[[], Any]] = None
        self.playing_song_changed_callback: Optional[Callable[[], Any]] = None
        self.api: NetEase = NetEase()
        self.playinfo_starts: float = time.time()

    @property
    def info(self) -> Dict[str, Any]:
        return self.storage.database['player_info']

    @property
    def songs(self) -> Dict[str, Dict[str, Any]]:
        return self.storage.database['songs']

    @property
    def index(self) -> int:
        return self.info['idx']

    @property
    def list(self) -> List[str]:
        return self.info['player_list']

    @property
    def order(self) -> List[int]:
        return self.info['playing_order']

    @property
    def mode(self) -> int:
        return self.info['playing_mode']

    @property
    def is_ordered_mode(self) -> bool:
        return self.mode == Player.MODE_ORDERED

    @property
    def is_ordered_loop_mode(self) -> bool:
        return self.mode == Player.MODE_ORDERED_LOOP

    @property
    def is_single_loop_mode(self) -> bool:
        return self.mode == Player.MODE_SINGLE_LOOP

    @property
    def is_random_mode(self) -> bool:
        return self.mode == Player.MODE_RANDOM

    @property
    def is_random_loop_mode(self) -> bool:
        return self.mode == Player.MODE_RANDOM_LOOP

    @property
    def config_notifier(self) -> Any:
        return self.config.get('notifier')

    @property
    def config_mpg123(self) -> List[str]:
        return self.config.get('mpg123_parameters')

    @property
    def current_song(self) -> Dict[str, Any]:
        if not self.songs:
            return {}
        if not self.is_index_valid:
            return {}
        song_id: str = self.list[self.index]
        return self.songs.get(song_id, {})

    @property
    def playing_id(self) -> Optional[str]:
        return self.current_song.get('song_id')

    @property
    def playing_name(self) -> Optional[str]:
        return self.current_song.get('song_name')

    @property
    def is_empty(self) -> bool:
        return len(self.list) == 0

    @property
    def is_index_valid(self) -> bool:
        return 0 <= self.index < len(self.list)

    def notify_playing(self) -> None:
        if not self.current_song:
            return
        if not self.config_notifier:
            return
        song: Dict[str, Any] = self.current_song
        notify('正在播放: {}\n{}-{}'.format(song['song_name'], song['artist'], song['album_name']))

    def notify_copyright_issue(self) -> None:
        log.warning('Song {} is unavailable due to copyright issue.'.format(self.playing_id))
        notify('版权限制，无法播放此歌曲')

    def change_mode(self, step: int = 1) -> None:
        self.info['playing_mode'] = (self.info['playing_mode'] + step) % 5

    def build_playinfo(self) -> None:
        if not self.current_song:
            return
        self.ui.build_playinfo(
            self.current_song['song_name'],
            self.current_song['artist'],
            self.current_song['album_name'],
            self.current_song['quality'],
            self.playinfo_starts,
            pause=not self.playing_flag
        )

    def add_songs(self, songs: List[Dict[str, Any]]) -> None:
        for song in songs:
            song_id: str = str(song['song_id'])
            self.info['player_list'].append(song_id)
            if song_id in self.songs:
                self.songs[song_id].update(song)
            else:
                self.songs[song_id] = song

    def refresh_urls(self) -> None:
        songs: List[Dict[str, Any]] = self.api.dig_info(self.list, 'refresh_urls')
        if songs:
            for song in songs:
                song_id: str = str(song['song_id'])
                if song_id in self.songs:
                    self.songs[song_id]['mp3_url'] = song['mp3_url']
                    self.songs[song_id]['expires'] = song['expires']
                    self.songs[song_id]['get_time'] = song['get_time']
                else:
                    self.songs[song_id] = song
            self.refresh_url_flag = True

    def stop(self) -> None:
        if not hasattr(self.popen_handler, 'poll') or self.popen_handler.poll():
            return
        self.playing_flag = False
        try:
            if not self.popen_handler.poll() and (not self.popen_handler.stdin.closed):
                self.popen_handler.stdin.write(b'Q\n')
                self.popen_handler.stdin.flush()
                self.popen_handler.communicate()
                self.popen_handler.kill()
        except Exception as e:
            log.warn(e)
        finally:
            for thread_i in range(0, len(self.MUSIC_THREADS) - 1):
                if self.MUSIC_THREADS[thread_i].is_alive():
                    try:
                        stop_thread(self.MUSIC_THREADS[thread_i])
                    except Exception as e:
                        log.warn(e)

    def tune_volume(self, up: int = 0) -> None:
        try:
            if self.popen_handler.poll():
                return
        except Exception as e:
            log.warn('Unable to tune volume: ' + str(e))
            return
        new_volume: int = self.info['playing_volume'] + up
        if new_volume < 0:
            new_volume = 0
        self.info['playing_volume'] = new_volume
        try:
            self.popen_handler.stdin.write('V {}\n'.format(self.info['playing_volume']).encode())
            self.popen_handler.stdin.flush()
        except Exception as e:
            log.warn(e)

    def switch(self) -> None:
        if not self.popen_handler:
            return
        if self.popen_handler.poll():
            return
        self.playing_flag = not self.playing_flag
        if not self.popen_handler.stdin.closed:
            self.popen_handler.stdin.write(b'P\n')
            self.popen_handler.stdin.flush()
        self.playinfo_starts = time.time()
        self.build_playinfo()

    def run_mpg123(
        self, 
        on_exit: Callable[[], Any], 
        url: str, 
        expires: int = -1, 
        get_time: int = -1
    ) -> None:
        para: List[str] = ['mpg123', '-R'] + self.config_mpg123
        self.popen_handler = subprocess.Popen(para, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not url:
            self.notify_copyright_issue()
            if not self.is_single_loop_mode:
                self.next()
            else:
                self.stop()
            return
        self.tune_volume()
        try:
            self.popen_handler.stdin.write(b'L ' + url.encode('utf-8') + b'\n')
            self.popen_handler.stdin.flush()
        except Exception:
            pass
        strout: str = ' '
        copyright_issue_flag: bool = False
        frame_cnt: int = 0
        while True:
            if not hasattr(self.popen_handler, 'poll') or self.popen_handler.poll():
                break
            if self.popen_handler.stdout.closed:
                break
            try:
                stroutlines: bytes = self.popen_handler.stdout.readline()
            except Exception as e:
                log.warn(e)
                break
            if not stroutlines:
                strout = ' '
                break
            else:
                strout_new: str = stroutlines.decode().strip()
                if strout_new[:2] != strout[:2]:
                    for thread_i in range(0, len(self.MUSIC_THREADS) - 1):
                        if self.MUSIC_THREADS[thread_i].is_alive():
                            try:
                                stop_thread(self.MUSIC_THREADS[thread_i])
                            except Exception as e:
                                log.warn(e)
                strout = strout_new
            if strout[:2] == '@F':
                out = strout.split(' ')
                frame_cnt += 1
                self.process_location = float(out[3])
                self.process_length = int(float(out[3]) + float(out[4]))
            elif strout[:2] == '@E':
                self.playing_flag = True
                if expires >= 0 and get_time >= 0 and (time.time() - expires - get_time >= 0):
                    self.refresh_urls()
                else:
                    copyright_issue_flag = True
                    self.notify_copyright_issue()
                break
            elif strout == '@P 0' and frame_cnt:
                self.playing_flag = True
                copyright_issue_flag = False
                break
            elif strout == '@P 0':
                self.playing_flag = True
                copyright_issue_flag = True
                self.notify_copyright_issue()
                break
        if self.playing_flag and self.refresh_url_flag:
            self.stop()
            self.playing_flag = True
            self.start_playing(lambda: 0, self.current_song)
            self.refresh_url_flag = False
        elif not self.playing_flag:
            self.stop()
        elif copyright_issue_flag and self.is_single_loop_mode:
            self.stop()
        else:
            self.next()

    def download_lyric(self, is_transalted: bool = False) -> None:
        key: str = 'lyric' if not is_transalted else 'tlyric'
        if key not in self.songs[str(self.playing_id)]:
            self.songs[str(self.playing_id)][key] = []
        if len(self.songs[str(self.playing_id)][key]) > 0:
            return
        if not is_transalted:
            lyric: Any = self.api.song_lyric(self.playing_id)
        else:
            lyric = self.api.song_tlyric(self.playing_id)
        self.songs[str(self.playing_id)][key] = lyric

    def download_song(self, song_id: Any, song_name: str, artist: str, url: str) -> None:

        def write_path(song_id_inner: Any, path: str) -> None:
            self.songs[str(song_id_inner)]['cache'] = path

        self.cache.add(song_id, song_name, artist, url, write_path)
        self.cache.start_download()

    def start_playing(self, on_exit: Callable[[], Any], args: Dict[str, Any]) -> threading.Thread:
        """
        Runs the given args in subprocess.Popen, and then calls the function
        on_exit when the subprocess completes.
        on_exit is a callable object, and args is a dictionary of parameters.
        """
        if 'cache' in args.keys() and os.path.isfile(args['cache']):
            thread: threading.Thread = threading.Thread(
                target=self.run_mpg123, 
                args=(on_exit, args['cache'])
            )
        else:
            thread = threading.Thread(
                target=self.run_mpg123,
                args=(on_exit, args['mp3_url'], args['expires'], args['get_time'])
            )
            cache_thread: threading.Thread = threading.Thread(
                target=self.download_song, 
                args=(args['song_id'], args['song_name'], args['artist'], args['mp3_url'])
            )
            cache_thread.start()
        thread.start()
        self.MUSIC_THREADS.append(thread)
        self.MUSIC_THREADS = [i for i in self.MUSIC_THREADS if i.is_alive()]
        lyric_download_thread: threading.Thread = threading.Thread(target=self.download_lyric)
        lyric_download_thread.start()
        tlyric_download_thread: threading.Thread = threading.Thread(target=self.download_lyric, args=(True,))
        tlyric_download_thread.start()
        return thread

    def replay(self) -> None:
        if not self.is_index_valid:
            self.stop()
            if self.end_callback:
                log.debug('Callback')
                self.end_callback()
            return
        if not self.current_song:
            return
        self.playing_flag = True
        self.playinfo_starts = time.time()
        self.build_playinfo()
        self.notify_playing()
        self.start_playing(lambda: 0, self.current_song)

    def shuffle_order(self) -> None:
        del self.order[:]
        self.order.extend(list(range(0, len(self.list))))
        random.shuffle(self.order)
        self.info['random_index'] = 0

    def new_player_list(self, type: Any, title: str, datalist: List[Dict[str, Any]], offset: int) -> None:
        self.info['player_list_type'] = type
        self.info['player_list_title'] = title
        self.info['player_list'] = []
        self.info['playing_order'] = []
        self.info['random_index'] = 0
        self.add_songs(datalist)

    def append_songs(self, datalist: List[Dict[str, Any]]) -> None:
        self.add_songs(datalist)

    def play_or_pause(self, idx: int, switch_flag: bool) -> None:
        if self.is_empty:
            return
        if self.index == idx and switch_flag:
            if not self.popen_handler:
                self.replay()
            else:
                self.switch()
        else:
            self.info['idx'] = idx
            self.stop()
            self.replay()

    def _swap_song(self) -> None:
        now_songs: int = self.order.index(self.index)
        self.order[0], self.order[now_songs] = self.order[now_songs], self.order[0]

    def _need_to_shuffle(self) -> bool:
        playing_order: List[int] = self.order
        random_index: int = self.info['random_index']
        if random_index >= len(playing_order) or playing_order[random_index] != self.index:
            return True
        else:
            return False

    def next_idx(self) -> None:
        if not self.is_index_valid:
            self.stop()
            return
        playlist_len: int = len(self.list)
        if self.mode == Player.MODE_ORDERED:
            if self.info['idx'] < playlist_len:
                self.info['idx'] += 1
        elif self.mode == Player.MODE_ORDERED_LOOP:
            self.info['idx'] = (self.index + 1) % playlist_len
        elif self.mode == Player.MODE_SINGLE_LOOP:
            self.info['idx'] = self.info['idx']
        else:
            playing_order_len: int = len(self.order)
            if self._need_to_shuffle():
                self.shuffle_order()
                self._swap_song()
                playing_order_len = len(self.order)
            self.info['random_index'] += 1
            if self.mode == Player.MODE_RANDOM_LOOP:
                self.info['random_index'] %= playing_order_len
            if self.info['random_index'] >= playing_order_len:
                self.info['idx'] = playlist_len
            else:
                self.info['idx'] = self.order[self.info['random_index']]
        if self.playing_song_changed_callback is not None:
            self.playing_song_changed_callback()

    def next(self) -> None:
        self.stop()
        self.next_idx()
        self.replay()

    def prev_idx(self) -> None:
        if not self.is_index_valid:
            self.stop()
            return
        playlist_len: int = len(self.list)
        if self.mode == Player.MODE_ORDERED:
            if self.info['idx'] > 0:
                self.info['idx'] -= 1
        elif self.mode == Player.MODE_ORDERED_LOOP:
            self.info['idx'] = (self.info['idx'] - 1) % playlist_len
        elif self.mode == Player.MODE_SINGLE_LOOP:
            self.info['idx'] = self.info['idx']
        else:
            playing_order_len: int = len(self.order)
            if self._need_to_shuffle():
                self.shuffle_order()
                playing_order_len = len(self.order)
            self.info['random_index'] -= 1
            if self.info['random_index'] < 0:
                if self.mode == Player.MODE_RANDOM:
                    self.info['random_index'] = 0
                else:
                    self.info['random_index'] %= playing_order_len
            self.info['idx'] = self.order[self.info['random_index']]
        if self.playing_song_changed_callback is not None:
            self.playing_song_changed_callback()

    def prev(self) -> None:
        self.stop()
        self.prev_idx()
        self.replay()

    def shuffle(self) -> None:
        self.stop()
        self.info['playing_mode'] = Player.MODE_RANDOM
        self.shuffle_order()
        self.info['idx'] = self.info['playing_order'][self.info['random_index']]
        self.replay()

    def volume_up(self) -> None:
        self.tune_volume(5)

    def volume_down(self) -> None:
        self.tune_volume(-5)

    def update_size(self) -> None:
        self.ui.update_size()
        self.build_playinfo()

    def cache_song(self, song_id: Any, song_name: str, artist: str, song_url: str) -> None:

        def on_exit(song_id_inner: Any, path: str) -> None:
            self.songs[str(song_id_inner)]['cache'] = path
            self.cache.enable = False

        self.cache.enable = True
        self.cache.add(song_id, song_name, artist, song_url, on_exit)
        self.cache.start_download()