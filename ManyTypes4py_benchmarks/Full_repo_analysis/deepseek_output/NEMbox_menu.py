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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
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
KEY_MAP: Dict[str, Union[int, str]] = Config().get('keymap')
COMMAND_LIST: List[int] = list(map(ord, KEY_MAP.values()))
if Config().get('mouse_movement'):
    KEY_MAP['mouseUp'] = 259
    KEY_MAP['mouseDown'] = 258
else:
    KEY_MAP['mouseUp'] = -259
    KEY_MAP['mouseDown'] = -258
shortcut: List[List[Union[str, int]]] = [[KEY_MAP['down'], 'Down', '下移'], [KEY_MAP['up'], 'Up', '上移'], ['<Num>+' + KEY_MAP['up'], '<num> Up', '上移num'], ['<Num>+' + KEY_MAP['down'], '<num> Down', '下移num'], [KEY_MAP['back'], 'Back', '后退'], [KEY_MAP['forward'], 'Forward', '前进'], [KEY_MAP['prevPage'], 'Prev page', '上一页'], [KEY_MAP['nextPage'], 'Next page', '下一页'], [KEY_MAP['search'], 'Search', '快速搜索'], [KEY_MAP['prevSong'], 'Prev song', '上一曲'], [KEY_MAP['nextSong'], 'Next song', '下一曲'], ['<Num>+' + KEY_MAP['nextSong'], '<Num> Next Song', '下num曲'], ['<Num>+' + KEY_MAP['prevSong'], '<Num> Prev song', '上num曲'], ['<Num>', 'Goto song <num>', '跳转指定歌曲id'], [KEY_MAP['playPause'], 'Play/Pause', '播放/暂停'], [KEY_MAP['shuffle'], 'Shuffle', '手气不错'], [KEY_MAP['volume+'], 'Volume+', '音量增加'], [KEY_MAP['volume-'], 'Volume-', '音量减少'], [KEY_MAP['menu'], 'Menu', '主菜单'], [KEY_MAP['presentHistory'], 'Present/History', '当前/历史播放列表'], [KEY_MAP['musicInfo'], 'Music Info', '当前音乐信息'], [KEY_MAP['playingMode'], 'Playing Mode', '播放模式切换'], [KEY_MAP['enterAlbum'], 'Enter album', '进入专辑'], [KEY_MAP['add'], 'Add', '添加曲目到打碟'], [KEY_MAP['djList'], 'DJ list', '打碟列表（退出后清空）'], [KEY_MAP['star'], 'Star', '添加到本地收藏'], [KEY_MAP['collection'], 'Collection', '本地收藏列表'], [KEY_MAP['remove'], 'Remove', '删除当前条目'], [KEY_MAP['moveDown'], 'Move Down', '向下移动当前条目'], [KEY_MAP['moveUp'], 'Move Up', '向上移动当前条目'], [KEY_MAP['like'], 'Like', '喜爱'], [KEY_MAP['cache'], 'Cache', '缓存歌曲到本地'], [KEY_MAP['nextFM'], 'Next FM', '下一 FM'], [KEY_MAP['trashFM'], 'Trash FM', '删除 FM'], [KEY_MAP['quit'], 'Quit', '退出'], [KEY_MAP['quitClear'], 'Quit&Clear', '退出并清除用户信息'], [KEY_MAP['help'], 'Help', '帮助'], [KEY_MAP['top'], 'Top', '回到顶部'], [KEY_MAP['bottom'], 'Bottom', '跳转到底部'], [KEY_MAP['countDown'], 'Count Down', '定时']]

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
        self.screen: Any = C.initscr()
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
        self.timer: Optional[Timer] = None
        self.key_list: List[int] = []
        self.pre_keylist: List[int] = []
        self.parser: Optional[Any] = None
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
        resp: Dict[str, Any] = self.api.login(account, md5pass)
        if resp['code'] == 200:
            userid: int = resp['account']['id']
            nickname: str = resp['profile']['nickname']
            self.storage.login(account, md5pass, userid, nickname)
            return True
        else:
            self.storage.logout()
            x: int = self.ui.build_login_error()
            if x >= 0 and C.keyname(x).decode('utf-8') != KEY_MAP['forward']:
                return False
            return self.login()

    def in_place_search(self) -> Tuple[List[Dict[str, Any]], str]:
        self.ui.screen.timeout(-1)
        prompt: str = '模糊搜索：'
        keyword: str = self.ui.get_param(prompt)
        if not keyword:
            return ([], '')
        if self.datalist == []:
            return ([], keyword)
        origin_index: int = 0
        for item in self.datalist:
            item['origin_index'] = origin_index
            origin_index += 1
        try:
            search_result: List[Tuple[Dict[str, Any], int]] = process.extract(keyword, self.datalist, limit=max(10, 2 * self.step))
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
        category_map: Dict[str, Any] = {'songs': SearchArg('搜索歌曲：', 1, lambda datalist: datalist), 'albums': SearchArg('搜索专辑：', 10, lambda datalist: datalist), 'artists': SearchArg('搜索艺术家：', 100, lambda datalist: datalist), 'playlists': SearchArg('搜索网易精选集：', 1000, lambda datalist: datalist), 'djRadios': SearchArg('搜索主播电台：', 1009, lambda datalist: datalist)}
        (prompt, api_type, post_process) = category_map[category]
        keyword: str = self.ui.get_param(prompt)
        if not keyword:
            return []
        data: Dict[str, Any] = self.api.search(keyword, api_type)
        if not data:
            return data
        datalist: List[Dict[str, Any]] = post_process(data.get(category, []))
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
        latest: Union[int, str] = Menu().check_version()
        if str(latest) > str(version) and latest != 0:
            notify('MusicBox Update == available', 1)
            time.sleep(0.5)
            notify('NetEase-MusicBox installed version:' + version + '\nNetEase-MusicBox latest version:' + latest, 0)

    def check_version(self) -> Union[int, str]:
        try:
            mobile: Dict[str, Any] = self.api.daily_task(is_mobile=True)
            pc: Dict[str, Any] = self.api.daily_task(is_mobile=False)
            if mobile['code'] == 200:
                notify('移动端签到成功', 1)
            if pc['code'] == 200:
                notify('PC端签到成功', 1)
            data: Dict[str, Any] = self.api.get_version()
            return data['info']['version']
        except KeyError:
            return 0

    def start_fork(self, version: str) -> None:
        pid: int = os.fork()
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
        datalist: List[Dict[str, Any]] = self.datalist
        offset: int = self.offset
        idx: int = self.index
        step: int = self.step
        if idx == offset:
            if offset == 0:
                return
            self.offset -= step
            self.index = offset - 1
        else:
            self.index = carousel(offset, min(len(datalist), offset + step) - 1, idx - 1)
        self.menu_starts = time.time()

    def down_key_event(self) -> None:
        datalist: List[Dict[str, Any]] = self.datalist
        offset: int = self.offset
        idx: int = self.index
        step: int = self.step
        if idx == min(len(datalist), offset + step) - 1:
            if offset + step >= len(datalist):
                return
            self.offset += step
            self.index = offset + step
        else:
            self.index = carousel(offset, min(len(datalist), offset + step) - 1, idx + 1)
        self.menu_starts = time.time()

    def space_key_event_in_search_result(self) -> None:
        origin_index: int = self.datalist[self.index]['origin_index']
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
            is_not_songs: bool = True
            self.player.play_or_pause(self.player.info['idx'], is_not_songs)
        self.build_menu_processbar()

    def space_key_event(self) -> None:
        idx: int = self.index
        datatype: str = self.datatype
        if not self.datalist:
            return
        if idx < 0 or idx >= len(self.datalist):
            self.player.info['idx'] = 0
        datatype_callback: Dict[str, Optional[Callable]] = {'songs': None, 'djprograms': None, 'fmsongs': self.fm_callback}
        if datatype in ['songs', 'djprograms', 'fmsongs']:
            self.player.new_player_list(datatype, self.title, self.datalist, -1)
            self.player.end_callback = datatype_callback[datatype]
            self.player.play_or_pause(idx, self.at_playing_list)
            self.at_playing_list = True
        else:
            is_not_songs: bool = True
            self.player.play_or_pause(self.player.info['idx'], is_not_songs)
        self.build_menu_processbar()

    def like_event(self) -> None:
        return_data: Optional[Any] = self.request_api(self.api.fm_like, self.player.playing_id)
        if return_data:
            song_name: str = self.player.playing_name
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
        idx: int = self.index
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
        datatype: str = self.datatype
        title: str = self.title
        datalist: List[Dict[str, Any]] = self.datalist
        offset: int = self.offset
        idx: int = self.index
        step: int = self.step
        if datatype == 'album':
            return
        if datatype in ['songs', 'fmsongs']:
            song_id: int = datalist[idx]['song_id']
            album_id: int = datalist[idx]['album_id']
            album_name: str = datalist[idx]['album_name']
        elif self.player.playing_flag:
            song_id: int = self.player.playing_id
            song_info: Dict[str, Any] = self.player.songs.get(str(song_id), {})
            album_id: int = song_info.get('album_id', 0)
            album_name: str = song_info.get('album_name', '')
        else:
            album_id: int = 0
        if album_id:
            self.stack.append