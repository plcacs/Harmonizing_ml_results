"""
网易云音乐 Menu
"""
import curses as C
import locale
import os
import signal
import sys
import threading
import time
import webbrowser
from collections import namedtuple
from copy import deepcopy
from threading import Timer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator, Set, Generator
from fuzzywuzzy import process
from . import logger
from .api import NetEase
from .cache import Cache
from .cmd_parser import cmd_parser
from .cmd_parser import erase_coroutine
from .cmd_parser import parse_keylist
from .config import Config
from .osdlyrics import pyqt_activity
from .osdlyrics import show_lyrics_new_process
from .osdlyrics import stop_lyrics_process
from .player import Player
from .storage import Storage
from .ui import Ui
from .utils import notify
locale.setlocale(locale.LC_ALL, '')
log = logger.getLogger(__name__)

def carousel(left: int, right: int, x: int) -> int:
    if x > right:
        return left
    elif x < left:
        return right
    else:
        return x
KEY_MAP = Config().get('keymap')
COMMAND_LIST = list(map(ord, KEY_MAP.values()))
if Config().get('mouse_movement'):
    KEY_MAP['mouseUp'] = 259
    KEY_MAP['mouseDown'] = 258
else:
    KEY_MAP['mouseUp'] = -259
    KEY_MAP['mouseDown'] = -258
shortcut: List[List[Union[int, str]]] = [[KEY_MAP['down'], 'Down', '下移'], [KEY_MAP['up'], 'Up', '上移'], ['<Num>+' + KEY_MAP['up'], '<num> Up', '上移num'], ['<Num>+' + KEY_MAP['down'], '<num> Down', '下移num'], [KEY_MAP['back'], 'Back', '后退'], [KEY_MAP['forward'], 'Forward', '前进'], [KEY_MAP['prevPage'], 'Prev page', '上一页'], [KEY_MAP['nextPage'], 'Next page', '下一页'], [KEY_MAP['search'], 'Search', '快速搜索'], [KEY_MAP['prevSong'], 'Prev song', '上一曲'], [KEY_MAP['nextSong'], 'Next song', '下一曲'], ['<Num>+' + KEY_MAP['nextSong'], '<Num> Next Song', '下num曲'], ['<Num>+' + KEY_MAP['prevSong'], '<Num> Prev song', '上num曲'], ['<Num>', 'Goto song <num>', '跳转指定歌曲id'], [KEY_MAP['playPause'], 'Play/Pause', '播放/暂停'], [KEY_MAP['shuffle'], 'Shuffle', '手气不错'], [KEY_MAP['volume+'], 'Volume+', '音量增加'], [KEY_MAP['volume-'], 'Volume-', '音量减少'], [KEY_MAP['menu'], 'Menu', '主菜单'], [KEY_MAP['presentHistory'], 'Present/History', '当前/历史播放列表'], [KEY_MAP['musicInfo'], 'Music Info', '当前音乐信息'], [KEY_MAP['playingMode'], 'Playing Mode', '播放模式切换'], [KEY_MAP['enterAlbum'], 'Enter album', '进入专辑'], [KEY_MAP['add'], 'Add', '添加曲目到打碟'], [KEY_MAP['djList'], 'DJ list', '打碟列表（退出后清空）'], [KEY_MAP['star'], 'Star', '添加到本地收藏'], [KEY_MAP['collection'], 'Collection', '本地收藏列表'], [KEY_MAP['remove'], 'Remove', '删除当前条目'], [KEY_MAP['moveDown'], 'Move Down', '向下移动当前条目'], [KEY_MAP['moveUp'], 'Move Up', '向上移动当前条目'], [KEY_MAP['like'], 'Like', '喜爱'], [KEY_MAP['cache'], 'Cache', '缓存歌曲到本地'], [KEY_MAP['nextFM'], 'Next FM', '下一 FM'], [KEY_MAP['trashFM'], 'Trash FM', '删除 FM'], [KEY_MAP['quit'], 'Quit', '退出'], [KEY_MAP['quitClear'], 'Quit&Clear', '退出并清除用户信息'], [KEY_MAP['help'], 'Help', '帮助'], [KEY_MAP['top'], 'Top', '回到顶部'], [KEY_MAP['bottom'], 'Bottom', '跳转到底部'], [KEY_MAP['countDown'], 'Count Down', '定时']]

class Menu(object):

    def __init__(self) -> None:
        self.quit: bool = False
        self.config: Config = Config()
        self.datatype: str = 'main'
        self.title: str = '网易云音乐'
        self.datalist: List[Dict[str, Any]] = [{'entry_name': '排行榜'}, {'entry_name': '艺术家'}, {'entry_name': '新碟上架'}, {'entry_name': '精选歌单'}, {'entry_name': '我的歌单'}, {'entry_name': '主播电台'}, {'entry_name': '每日推荐歌曲'}, {'entry_name': '每日推荐歌单'}, {'entry_name': '私人FM'}, {'entry_name': '搜索'}, {'entry_name': '帮助'}]
        self.offset: int = 0
        self.index: int = 0
        self.storage: Storage = Storage()
        self.storage.load()
        self.collection: List[Dict[str, Any]] = self.storage.database['collections']
        self.player: Player = Player()
        self.player.playing_song_changed_callback = self.song_changed_callback
        self.cache: Cache = Cache()
        self.ui: Ui = Ui()
        self.api: NetEase = NetEase()
        self.screen = C.initscr()
        self.screen.keypad(1)
        self.step: int = Config().get('page_length')
        if self.step == 0:
            self.step = max(int(self.ui.y * 4 / 5) - 10, 1)
        self.stack: List[List[Any]] = []
        self.djstack: List[Dict[str, Any]] = []
        self.at_playing_list: bool = False
        self.enter_flag: bool = True
        signal.signal(signal.SIGWINCH, self.change_term)
        signal.signal(signal.SIGINT, self.send_kill)
        signal.signal(signal.SIGTERM, self.send_kill)
        self.menu_starts: float = time.time()
        self.countdown_start: float = time.time()
        self.countdown: int = -1
        self.is_in_countdown: bool = False
        self.timer: Optional[Timer] = 0
        self.key_list: List[int] = []
        self.pre_keylist: List[int] = []
        self.parser: Optional[Generator[None, int, None]] = None
        self.at_search_result: bool = False

    @property
    def user(self) -> Dict[str, Any]:
        return self.storage.database['user']

    @property
    def account(self) -> str:
        return self.user['username']

    @property
    def md5pass(self) -> str:
        return self.user['password']

    @property
    def userid(self) -> int:
        return self.user['user_id']

    @property
    def username(self) -> str:
        return self.user['nickname']

    def login(self) -> bool:
        if self.account and self.md5pass:
            (account, md5pass) = (self.account, self.md5pass)
        else:
            (account, md5pass) = self.ui.build_login()
        resp = self.api.login(account, md5pass)
        if resp['code'] == 200:
            userid = resp['account']['id']
            nickname = resp['profile']['nickname']
            self.storage.login(account, md5pass, userid, nickname)
            return True
        else:
            self.storage.logout()
            x = self.ui.build_login_error()
            if x >= 0 and C.keyname(x).decode('utf-8') != KEY_MAP['forward']:
                return False
            return self.login()

    def in_place_search(self) -> Tuple[List[Dict[str, Any]], str]:
        self.ui.screen.timeout(-1)
        prompt = '模糊搜索：'
        keyword = self.ui.get_param(prompt)
        if not keyword:
            return ([], '')
        if self.datalist == []:
            return ([], keyword)
        origin_index = 0
        for item in self.datalist:
            item['origin_index'] = origin_index
            origin_index += 1
        try:
            search_result = process.extract(keyword, self.datalist, limit=max(10, 2 * self.step))
            if not search_result:
                return (search_result, keyword)
            search_result.sort(key=lambda ele: ele[1], reverse=True)
            return (list(map(lambda ele: ele[0], search_result)), keyword)
        except Exception as e:
            log.warn(e)
            return ([], keyword)

    def search(self, category: str) -> List[Dict[str, Any]]:
        self.ui.screen.timeout(-1)
        SearchArg = namedtuple('SearchArg', ['prompt', 'api_type', 'post_process'])
        category_map: Dict[str, SearchArg] = {'songs': SearchArg('搜索歌曲：', 1, lambda datalist: datalist), 'albums': SearchArg('搜索专辑：', 10, lambda datalist: datalist), 'artists': SearchArg('搜索艺术家：', 100, lambda datalist: datalist), 'playlists': SearchArg('搜索网易精选集：', 1000, lambda datalist: datalist), 'djRadios': SearchArg('搜索主播电台：', 1009, lambda datalist: datalist)}
        (prompt, api_type, post_process) = category_map[category]
        keyword = self.ui.get_param(prompt)
        if not keyword:
            return []
        data = self.api.search(keyword, api_type)
        if not data:
            return data
        datalist = post_process(data.get(category, []))
        return self.api.dig_info(datalist, category)

    def change_term(self, signum: int, frame: Any) -> None:
        self.ui.screen.clear()
        self.ui.screen.refresh()

    def send_kill(self, signum: int, fram: Any) -> None:
        if pyqt_activity:
            stop_lyrics_process()
        self.player.stop()
        self.cache.quit()
        self.storage.save()
        C.endwin()
        sys.exit()

    def update_alert(self, version: str) -> None:
        latest = Menu().check_version()
        if str(latest) > str(version) and latest != 0:
            notify('MusicBox Update == available', 1)
            time.sleep(0.5)
            notify('NetEase-MusicBox installed version:' + version + '\nNetEase-MusicBox latest version:' + latest, 0)

    def check_version(self) -> Union[str, int]:
        try:
            mobile = self.api.daily_task(is_mobile=True)
            pc = self.api.daily_task(is_mobile=False)
            if mobile['code'] == 200:
                notify('移动端签到成功', 1)
            if pc['code'] == 200:
                notify('PC端签到成功', 1)
            data = self.api.get_version()
            return data['info']['version']
        except KeyError:
            return 0

    def start_fork(self, version: str) -> None:
        pid = os.fork()
        if pid == 0:
            Menu().update_alert(version)
        else:
            Menu().start()

    def next_song(self) -> None:
        if self.player.is_empty:
            return
        self.player.next()

    def previous_song(self) -> None:
        if self.player.is_empty:
            return
        self.player.prev()

    def prev_key_event(self) -> None:
        self.player.prev_idx()

    def next_key_event(self) -> None:
        self.player.next_idx()

    def up_key_event(self) -> None:
        datalist = self.datalist
        offset = self.offset
        idx = self.index
        step = self.step
        if idx == offset:
            if offset == 0:
                return
            self.offset -= step
            self.index = offset - 1
        else:
            self.index = carousel(offset, min(len(datalist), offset + step) - 1, idx - 1)
        self.menu_starts = time.time()

    def down_key_event(self) -> None:
        datalist = self.datalist
        offset = self.offset
        idx = self.index
        step = self.step
        if idx == min(len(datalist), offset + step) - 1:
            if offset + step >= len(datalist):
                return
            self.offset += step
            self.index = offset + step
        else:
            self.index = carousel(offset, min(len(datalist), offset + step) - 1, idx + 1)
        self.menu_starts = time.time()

    def space_key_event_in_search_result(self) -> None:
        origin_index = self.datalist[self.index]['origin_index']
        (datatype, title, datalist, offset, index) = self.stack[-1]
        if datatype == 'songs':
            self.player.new_player_list('songs', title, datalist, -1)
            self.player.end_callback = None
            self.player.play_or_pause(origin_index, self.at_playing_list)
            self.at_playing_list = False
        elif datatype == 'djprograms':
            self.player.new_player_list('djprograms', title, datalist, -1)
            self.player.end_callback = None
            self.player.play_or_pause(origin_index, self.at_playing_list)
            self.at_playing_list = False
        elif datatype == 'fmsongs':
            self.player.change_mode(0)
            self.player.new_player_list('fmsongs', title, datalist, -1)
            self.player.end_callback = self.fm_callback
            self.player.play_or_pause(origin_index, self.at_playing_list)
            self.at_playing_list = False
        else:
            is_not_songs = True
            self.player.play_or_pause(self.player.info['idx'], is_not_songs)
        self.build_menu_processbar()

    def space_key_event(self) -> None:
        idx = self.index
        datatype = self.datatype
        if not self.datalist:
            return
        if idx < 0 or idx >= len(self.datalist):
            self.player.info['idx'] = 0
        datatype_callback: Dict[str, Optional[Callable[[], None]]] = {'songs': None, 'djprograms': None, 'fmsongs': self.fm_callback}
        if datatype in ['songs', 'djprograms', 'fmsongs']:
            self.player.new_player_list(datatype, self.title, self.datalist, -1)
            self.player.end_callback = datatype_callback[datatype]
            self.player.play_or_pause(idx, self.at_playing_list)
            self.at_playing_list = True
        else:
            is_not_songs = True
            self.player.play_or_pause(self.player.info['idx'], is_not_songs)
        self.build_menu_processbar()

    def like_event(self) -> None:
        return_data = self.request_api(self.api.fm_like, self.player.playing_id)
        if return_data:
            song_name = self.player.playing_name
            notify('%s added successfully!' % song_name, 0)
        else:
            notify('Adding song failed!', 0)

    def back_page_event(self) -> None:
        if len(self.stack) == 1:
            return
        self.menu_starts = time.time()
        (self.datatype, self.title, self.datalist, self.offset, self.index) = self.stack.pop()
        self.at_playing_list = False
        self.at_search_result = False

    def enter_page_event(self) -> None:
        idx = self.index
        self.enter_flag = True
        if len(self.datalist) <= 0:
            return
        if self.datatype == 'comments':
            return
        self.menu_starts = time.time()
        self.ui.build_loading()
        self.dispatch_enter(idx)
        if self.enter_flag:
            self.index = 0
            self.offset = 0

    def album_key_event(self) -> None:
        datatype = self.datatype
        title = self.title
        datalist = self.datalist
        offset = self.offset
        idx = self.index
        step = self.step
        if datatype == 'album':
            return
        if datatype in ['songs', 'fmsongs']:
            song_id = datalist[idx]['song_id']
            album_id = datalist[idx]['album_id']
            album_name = datalist[idx]['album_name']
        elif self.player.playing_flag:
            song_id = self.player.playing_id
            song_info = self.player.songs.get(str(song_id), {})
            album_id = song_info.get('album_id', '')
            album_name = song_info.get('album_name', '')
        else:
            album_id = 0
        if album_id:
            self.stack.append([datatype, title, datalist, offset, self.index])
            songs = self.api.album(album_id)
            self.datatype = 'songs'
            self.datalist = self.api.dig_info(songs, 'songs')
            self.title = '网易云音乐 > 专辑 > %s' % album_name
            for i in range(len(self.datalist)):
                if self.datalist[i]['song_id'] == song_id:
                    self.offset = i - i % step
                    self.index = i
                    return
        self.build_menu_processbar()

    def num_jump_key_event(self) -> None:
        result = parse_keylist(self.key_list)
        (num, cmd) = result
        if num == 0:
            num = 1
        for _ in range(num):
            if cmd in (KEY_MAP['mouseUp'], ord(KEY_MAP['up'])):
                self.up_key_event()
            elif cmd in (KEY_MAP['mouseDown'], ord(KEY_MAP['down'])):
                self.down_key_event()
            elif cmd == ord(KEY_MAP['nextSong']):
                self.next_key_event()
            elif cmd == ord(KEY_MAP['prevSong']):
                self.prev_key_event()
        if cmd in (ord(KEY_MAP['nextSong']), ord(KEY_MAP['prevSong'])):
            self.player.stop()
            self.player.replay()
        self.build_menu_processbar()

    def digit_key_song_event(self) -> None:
        """直接跳到指定id 歌曲"""
        step = self.step
        self.key_list.pop()
        song_index = parse_keylist(self.key_list)
        if self.index != song_index:
            self.index = song_index
            self.offset = self.index - self.index % step
            self.build_menu_processbar()
            self.ui.screen.refresh()

    def time_key_event(self) -> None:
        self.countdown_start = time.time()
        countdown = self.ui.build_timing()
        if not countdown.isdigit():
            notify('The input should be digit')
        countdown = int(countdown)
        if countdown > 0:
            notify('The musicbox will exit in {} minutes'.format(countdown))
            self.countdown = countdown * 60
            self.is_in_countdown = True
            self.timer = Timer(self.countdown, self.stop, ())
            self.timer.start()
        else:
            notify('The timing exit has been canceled')
            self.is_in_countdown = False
            if self.timer:
                self.timer.cancel()
        self.build_menu_processbar()

    def down_page_event(self) -> None:
        offset = self.offset
        datalist = self.datalist
        step = self.step
        if offset + step >= len(datalist):
            return
        self.menu_starts = time.time()
        self.offset += step
        self.index = (self.index + step) // step * step

    def up_page_event(self) -> None:
        offset = self.offset
        step = self.step
        if offset == 0:
            return
        self.menu_starts = time.time()
        self.offset -= step
        self.index = (self.index - step) // step * step

    def resize_key_event(self) -> None:
        self.player.update_size()

    def build_menu_processbar(self) -> None:
        self.ui.build_process_bar(self.player.current_song, self.player.process_location, self.player.process_length, self.player.playing_flag, self.player.info['playing_mode'])
        self.ui.build_menu(self.datatype, self.title, self.datalist, self.offset, self.index, self.step, self.menu_starts)

    def quit_event(self) -> None:
        self.config.save_config_file()
        sys.exit(0)

    def stop(self) -> None:
        self.quit = True
        self.player.stop()
        self.cache.quit()
        self.storage.save()
        C.endwin()

    def start(self) -> None:
        self.menu_starts = time.time()
        self.ui.build_menu(self.datatype, self.title, self.datalist, self.offset, self.index, self.step, self.menu_starts)
        self.stack.append([self.datatype, self.title, self.datalist, self.offset, self.index])
        if pyqt_activity:
            show_lyrics_new_process()
        pre_key = -1
        keylist = self.key_list
        self.parser = cmd_parser(keylist)
        erase_cmd_list: List[int] = []
        erase_coro = erase_coroutine(erase_cmd_list)
        next(self.parser)
        next(erase_coro)
        while not self.quit:
            datatype = self.datatype
            title = self.title
            datalist = self.datalist
            offset = self.offset
            idx = self.index
            step = self.step
            self.screen.timeout(self.config.get('input_timeout'))
            key = self.screen.getch()
            if key in COMMAND_LIST and key != ord(KEY_MAP['nextSong']) and (key != ord(KEY_MAP['prevSong'])):
                if not set(self.pre_keylist) | {ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong'])} == {ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong'])}:
                    self.pre_keylist.append(key)
                self.key_list = deepcopy(self.pre_keylist)
                self.pre_keylist.clear()
            elif key in range(48, 58) or key == ord(KEY_MAP['nextSong']) or key == ord(KEY_MAP['prevSong']):
                self.pre_keylist.append(key)
            elif key == -1 and (pre_key == ord(KEY_MAP['nextSong']) or pre_key == ord(KEY_MAP['prevSong'])):
                self.key_list = deepcopy(self.pre_keylist)
                self.pre_keylist.clear()
            elif key == 27:
                self.pre_keylist.clear()
                self.key_list.clear()
            keylist = self.key_list
            if keylist and set(keylist) | set(range(48, 58)) | {ord(KEY_MAP['jumpIndex'])} == set(range(48, 58)) | {ord(KEY_MAP['jumpIndex'])}:
                self.digit_key_song_event()
                self.key_list.clear()
                continue
            if len(keylist) > 0 and set(keylist) | {ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong'])} == {ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong'])}:
                self.player.stop()
                self.player.replay()
                self.key_list.clear()
                continue
            if len(keylist) > 1:
                if parse_keylist(keylist):
                    self.num_jump_key_event()
                    self.key_list.clear()
                    continue
            if key == -1:
                self.player.update_size()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['quit']:
                if pyqt_activity:
                    stop_lyrics_process()
                break
            elif C.keyname(key).decode('utf-8') == KEY_MAP['quitClear']:
                if pyqt_activity:
                    stop_lyrics_process()
                self.api.logout()
                break
            elif C.keyname(key).decode('utf-8') == KEY_MAP['up'] and pre_key not in range(ord('0'), ord('9')):
                self.up_key_event()
            elif self.config.get('mouse_movement') and key == KEY_MAP['mouseUp']:
                self.up_key_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['down'] and pre_key not in range(ord('0'), ord('9')):
                self.down_key_event()
            elif self.config.get('mouse_movement') and key == KEY_MAP['mouseDown']:
                self.down_key_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['prevPage']:
                self.up_page_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['nextPage']:
                self.down_page_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['forward'] or key == 10:
                self.enter_page_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['back']:
                self.back_page_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['search']:
                if self.at_search_result:
                    self.back_page_event()
                self.stack.append([self.datatype, self.title, self.datalist, self.offset, self.index])
                (self.datalist, keyword) = self.in_place_search()
                self.title += ' > ' + keyword + ' 的搜索结果'
                self.offset = 0
                self.index = 0
                self.at_search_result = True
            elif C.keyname(key).decode('utf-8') == KEY_MAP['nextSong'] and pre_key not in range(ord('0'), ord('9')):
                self.next_key_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['prevSong'] and pre_key not in range(ord('0'), ord('9')):
                self.prev_key_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['volume+']:
                self.player.volume_up()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['volume-']:
                self.player.volume_down()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['shuffle']:
                if len(self.player.info['player_list']) == 0:
                    continue
                self.player.shuffle()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['like']:
                return_data = self.request_api(self.api.fm_like, self.player.playing_id)
                if return_data:
                    song_name = self.player.playing_name
                    notify('%s added successfully!' % song_name, 0)
                else:
                    notify('Adding song failed!', 0)
            elif C.keyname(key).decode('utf-8') == KEY_MAP['trashFM']:
                if self.datatype == 'fmsongs':
                    if len(self.player.info['player_list']) == 0:
                        continue
                    self.player.next()
                    return_data = self.request_api(self.api.fm_trash, self.player.playing_id)
                    if return_data:
                        notify('Deleted successfully!', 0)
            elif C.keyname(key).decode('utf-8') == KEY_MAP['nextFM']:
                if self.datatype == 'fmsongs':
                    if self.player.end_callback:
                        self.player.end_callback()
                    else:
                        self.datalist.extend(self.get_new_fm())
                self.build_menu_processbar()
                self.index = len(self.datalist) - 1
                self.offset = self.index - self.index % self.step
            elif C.keyname(key).decode('utf-8') == KEY_MAP['playPause']:
                if self.at_search_result:
                    self.space_key_event_in_search_result()
                else:
                    self.space_key_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['presentHistory']:
                self.show_playing_song()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['playingMode']:
                self.player.change_mode()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['enterAlbum']:
                if datatype == 'album':
                    continue
                if datatype in ['songs', 'fmsongs']:
                    song_id = datalist[idx]['song_id']
                    album_id = datalist[idx]['album_id']
                    album_name = datalist[idx]['album_name']
                elif self.player.playing_flag:
                    song_id = self.player.playing_id
                    song_info = self.player.songs.get(str(song_id), {})
                    album_id = song_info.get('album_id', '')
                    album_name = song_info.get('album_name', '')
                else:
                    album_id = 0
                if album_id:
                    self.stack.append([datatype, title, datalist, offset, self.index])
                    songs = self.api.album(album_id)
                    self.datatype = 'songs'
                    self.datalist = self.api.dig_info(songs, 'songs')
                    self.title = '网易云音乐 > 专辑 > %s' % album_name
                    for i in range(len(self.datalist)):
                        if self.datalist[i]['song_id'] == song_id:
                            self.offset = i - i % step
                            self.index = i
                            break
            elif C.keyname(key).decode('utf-8') == KEY_MAP['add']:
                if (self.datatype == 'songs' or self.datatype == 'djprograms') and len(self.datalist) != 0:
                    self.djstack.append(datalist[idx])
                elif datatype == 'artists':
                    pass
            elif C.keyname(key).decode('utf-8') == KEY_MAP['djList']:
                self.stack.append([self.datatype, self.title, self.datalist, self.offset, self.index])
                self.datatype = 'songs'
                self.title = '网易云音乐 > 打碟'
                self.datalist = self.djstack
                self.offset = 0
                self.index = 0
            elif C.keyname(key).decode('utf-8') == KEY_MAP['star']:
                if (self.datatype == 'songs' or self.datatype == 'djprograms') and len(self.datalist) != 0:
                    self.collection.append(self.datalist[self.index])
                    notify('Added successfully', 0)
            elif C.keyname(key).decode('utf-8') == KEY_MAP['collection']:
                self.stack.append([self.datatype, self.title, self.datalist, self.offset, self.index])
                self.datatype = 'songs'
                self.title = '网易云音乐 > 本地收藏'
                self.datalist = self.collection
                self.offset = 0
                self.index = 0
            elif C.keyname(key).decode('utf-8') == KEY_MAP['remove']:
                if self.datatype in ('songs', 'djprograms', 'fmsongs') and len(self.datalist) != 0:
                    self.datalist.pop(self.index)
                    log.warn(self.index)
                    log.warn(len(self.datalist))
                    if self.index == len(self.datalist):
                        self.up_key_event()
                    self.index = carousel(self.offset, min(len(self.datalist), self.offset + self.step) - 1, self.index)
            elif C.keyname(key).decode('utf-8') == KEY_MAP['countDown']:
                self.time_key_event()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['moveDown']:
                if self.datatype != 'main' and len(self.datalist) != 0 and (self.index + 1 != len(self.datalist)):
                    self.menu_starts = time.time()
                    song = self.datalist.pop(self.index)
                    self.datalist.insert(self.index + 1, song)
                    self.index = self.index + 1
                    if self.index >= self.offset + self.step:
                        self.offset = self.offset + self.step
            elif C.keyname(key).decode('utf-8') == KEY_MAP['moveUp']:
                if self.datatype != 'main' and len(self.datalist) != 0 and (self.index != 0):
                    self.menu_starts = time.time()
                    song = self.datalist.pop(self.index)
                    self.datalist.insert(self.index - 1, song)
                    self.index = self.index - 1
                    if self.index < self.offset:
                        self.offset = self.offset - self.step
            elif C.keyname(key).decode('utf-8') == KEY_MAP['menu']:
                if self.datatype != 'main':
                    self.stack.append([self.datatype, self.title, self.datalist, self.offset, self.index])
                    (self.datatype, self.title, self.datalist, *_) = self.stack[0]
                    self.offset = 0
                    self.index = 0
            elif C.keyname(key).decode('utf-8') == KEY_MAP['top']:
                if self.datatype == 'help':
                    webbrowser.open_new_tab('https://github.com/darknessomi/musicbox')
                else:
                    self.index = 0
                    self.offset = 0
            elif C.keyname(key).decode('utf-8') == KEY_MAP['bottom']:
                self.index = len(self.datalist) - 1
                self.offset = self.index - self.index % self.step
            elif C.keyname(key).decode('utf-8') == KEY_MAP['cache']:
                s = self.datalist[self.index]
                cache_thread = threading.Thread(target=self.player.cache_song, args=(s['song_id'], s['song_name'], s['artist'], s['mp3_url']))
                cache_thread.start()
            elif C.keyname(key).decode('utf-8') == KEY_MAP['musicInfo']:
                if self.player.playing_id != -1:
                    webbrowser.open_new_tab('http://music.163.com/song?id={}'.format(self.player.playing_id))
            self.player.update_size()
            pre_key = key
            self.ui.screen.refresh()
            self.ui.update_size()
            current_step = max(int(self.ui.y * 4 / 5) - 10, 1)
            if self.step != current_step and self.config.get('page_length') == 0:
                self.step = current_step
                self.index = 0
            self.build_menu_processbar()
        self.stop()

    def dispatch_enter(self, idx: int) -> bool:
        netease = self.api
        datatype = self.datatype
        title = self.title
        datalist = self.datalist
        offset = self.offset
        index = self.index
        self.stack.append([datatype, title, datalist, offset, index])
        if idx >= len(self.datalist):
            return False
        if datatype == 'main':
            self.choice_channel(idx)
        elif datatype == 'artists':
            artist_name = datalist[idx]['artists_name']
            artist_id = datalist[idx]['artist_id']
            self.datatype = 'artist_info'
            self.title += ' > ' + artist_name
            self.datalist = [{'item': '{}的热门歌曲'.format(artist_name), 'id': artist_id}, {'item': '{}的所有专辑'.format(artist_name), 'id': artist_id}]
        elif datatype == 'artist_info':
            self.title += ' > ' + datalist[idx]['item']
            artist_id = datalist[0]['id']
            if idx == 0:
                self.datatype = 'songs'
                songs = netease.artists(artist_id)
                self.datalist = netease.dig_info(songs, 'songs')
            elif idx == 1:
                albums = netease.get_artist_album(artist_id)
                self.datatype = 'albums'
                self.datalist = netease.dig_info(albums, 'albums')
        elif datatype == 'djRadios':
            radio_id = datalist[idx]['id']
            programs = netease.alldjprograms(radio_id)
            self.title += ' > ' + datalist[idx]['name']
            self.datatype = 'djprograms'
            self.datalist = netease.dig_info(programs, 'djprograms')
        elif datatype == 'albums':
            album_id = datalist[idx]['album_id']
            songs = netease.album(album_id)
            self.datatype = 'songs'
            self.datalist = netease.dig_info(songs, 'songs')
            self.title += ' > ' + datalist[idx]['albums_name']
        elif datatype == 'recommend_lists':
            data = self.datalist[idx]
            self.datatype = data['datatype']
            self.datalist = netease.dig_info(data['callback'](), self.datatype)
            self.title += ' > ' + data['title']
        elif datatype in ['top_playlists', 'playlists']:
            playlist_id = datalist[idx]['playlist_id']
            songs = netease.playlist_songlist(playlist_id)
            self.datatype = 'songs'
            self.datalist = netease.dig_info(songs, 'songs')
            self.title += ' > ' + datalist[idx]['playlist_name']
        elif datatype == 'playlist_classes':
            data = self.datalist[idx]
            self.datatype = 'playlist_class_detail'
            self.datalist = netease.dig_info(data, self.datatype)
            self.title += ' > ' + data
        elif datatype == 'playlist_class_detail':
            data = self.datalist[idx]
            self.datatype = 'top_playlists'
            self.datalist = netease.dig_info(netease.top_playlists(data), self.datatype)
            self.title += ' > ' + data
        elif datatype in ['songs', 'fmsongs']:
            song_id = datalist[idx]['song_id']
            comments = self.api.song_comments(song_id, limit=100)
            try:
                hotcomments = comments['hotComments']
                comcomments = comments['comments']
            except KeyError:
                hotcomments = comcomments = []
            self.datalist = []
            for one_comment in hotcomments:
                self.datalist.append({'comment_content': '(热评 %s❤️ ️) %s: %s' % (one_comment['likedCount'], one_comment['user']['nickname'], one_comment['content'])})
            for one_comment in comcomments:
                self.datalist.append({'comment_content': '(%s❤️ ️) %s: %s' % (one_comment['likedCount'], one_comment['user']['nickname'], one_comment['content'])})
            self.datatype = 'comments'
            self.title = '网易云音乐 > 评论: %s' % datalist[idx]['song_name']
            self.offset = 0
            self.index = 0
        elif datatype == 'toplists':
            songs = netease.top_songlist(idx)
            self.title += ' > ' + self.datalist[idx]
            self.datalist = netease.dig_info(songs, 'songs')
            self.datatype = 'songs'
        elif datatype == 'search':
            self.index = 0
            self.offset = 0
            SearchCategory = namedtuple('SearchCategory', ['type', 'title'])
            idx_map = {0: SearchCategory('playlists', '精选歌单搜索列表'), 1: SearchCategory('songs', '歌曲搜索列表'), 2: SearchCategory('artists', '艺术家搜索列表'), 3: SearchCategory('albums', '专辑搜索列表'), 4: SearchCategory('djRadios', '主播电台搜索列表')}
            (self.datatype, self.title) = idx_map[idx]
            self.datalist = self.search(self.datatype)
        else:
            self.enter_flag = False
        return True

    def show_playing_song(self) -> None:
        if self.player.is_empty:
            return
        if not self.at_playing_list and (not self.at_search_result):
            self.stack.append([self.datatype, self.title, self.datalist, self.offset, self.index])
            self.at_playing_list = True
        if self.at_search_result:
            self.back_page_event()
        self.datatype = self.player.info['player_list_type']
        self.title = self.player.info['player_list_title']
        self.datalist = [self.player.songs[i] for i in self.player.info['player_list']]
        self.index = self.player.info['idx']
        self.offset = self.index // self.step * self.step

    def song_changed_callback(self) -> None:
        if self.at_playing_list:
            self.show_playing_song()

    def fm_callback(self) -> None:
        data = self.get_new_fm()
        self.player.append_songs(data)
        if self.datatype == 'fmsongs':
            if self.player.is_empty:
                return
            self.datatype = self.player.info['player_list_type']
            self.title = self.player.info['player_list_title']
            self.datalist = []
            for i in self.player.info['player_list']:
                self.datalist.append(self.player.songs[i])
            self.index = self.player.info['idx']
            self.offset = self.index // self.step * self.step
            if not self.player.playing_flag:
                switch_flag = False
                self.player.play_or_pause(self.index, switch_flag)

    def request_api(self, func: Callable, *args: Any) -> Any:
        result = func(*args)
        if result:
            return result
        if not self.login():
            notify('You need to log in')
            return False
        return func(*args)

    def get_new_fm(self) -> List[Dict[str, Any]]:
        data = self.request_api(self.api.personal_fm)
        if not data:
            return []
        return self.api.dig_info(data, 'fmsongs')

    def choice_channel(self, idx: int) -> None:
        self.offset = 0
        self.index = 0
        if idx == 0:
            self.datalist = self.api.toplists
            self.title += ' > 排行榜'
            self.datatype = 'toplists'
        elif idx == 1:
            artists = self.api.top_artists()
            self.datalist = self.api.dig_info(artists, 'artists')
            self.title += ' > 艺术家'
            self.datatype = 'artists'
        elif idx == 2:
            albums = self.api.new_albums()
            self.datalist = self.api.dig_info(albums, 'albums')
            self.title += ' > 新碟上架'
            self.datatype = 'albums'
        elif idx == 3:
            self.datalist = [{'title': '全站置顶', 'datatype': 'top_playlists', 'callback': self.api.top_playlists}, {'title': '分类精选', 'datatype': 'playlist_classes', 'callback': lambda : []}]
            self.title += ' > 精选歌单'
            self.datatype = 'recommend_lists'
        elif idx == 4:
            if not self.account:
                self.login()
            myplaylist = self.request_api(self.api.user_playlist, self.userid)
            self.datatype = 'top_playlists'
            self.datalist = self.api.dig_info(myplaylist, self.datatype)
            self.title += ' > ' + self.username + ' 的歌单'
        elif idx == 5:
            self.datatype = 'djRadios'
            self.title += ' > 主播电台'
            self.datalist = self.api.djRadios()
        elif idx == 6:
            self.datatype = 'songs'
            self.title += ' > 每日推荐歌曲'
            myplaylist = self.request_api(self.api.recommend_playlist)
            if myplaylist == -1:
                return
            self.datalist = self.api.dig_info(myplaylist, self.datatype)
        elif idx == 7:
            myplaylist = self.request_api(self.api.recommend_resource)
            self.datatype = 'top_playlists'
            self.title += ' > 每日推荐歌单'
            self.datalist = self.api.dig_info(myplaylist, self.datatype)
        elif idx == 8:
            self.datatype = 'fmsongs'
            self.title += ' > 私人FM'
            self.datalist = self.get_new_fm()
        elif idx == 9:
            self.datatype = 'search'
            self.title += ' > 搜索'
            self.datalist = ['歌曲', '艺术家', '专辑', '主播电台', '网易精选集']
        elif idx == 10:
            self.datatype = 'help'
            self.title += ' > 帮助'
            self.datalist = shortcut
