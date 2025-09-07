# Merged Python files from musicbox
# Total files: 20


# File: NEMbox\__init__.py

"""
__   ___________________________________________
| \\  ||______   |   |______|_____||______|______
|  \\_||______   |   |______|     |______||______

________     __________________________  _____ _     _
|  |  ||     ||______  |  |      |_____]|     | \\___/
|  |  ||_____|______|__|__|_____ |_____]|_____|_/   \\_


+ ------------------------------------------ +
|   NetEase-MusicBox               320kbps   |
+ ------------------------------------------ +
|                                            |
|   ++++++++++++++++++++++++++++++++++++++   |
|   ++++++++++++++++++++++++++++++++++++++   |
|   ++++++++++++++++++++++++++++++++++++++   |
|   ++++++++++++++++++++++++++++++++++++++   |
|   ++++++++++++++++++++++++++++++++++++++   |
|                                            |
|   A sexy cli musicbox based on Python      |
|   Music resource from music.163.com        |
|                                            |
|   Built with love to music by omi          |
|                                            |
+ ------------------------------------------ +

"""
from importlib_metadata import version
from .const import Constant
from .utils import create_dir
from .utils import create_file
__version__ = version('NetEase-MusicBox')

def create_config():
    create_dir(Constant.conf_dir)
    create_dir(Constant.download_dir)
    create_file(Constant.storage_path)
    create_file(Constant.log_path, default='')
    create_file(Constant.cookie_path, default='# Netscape HTTP Cookie File\n')
create_config()

# File: NEMbox\__main__.py

import _curses
import argparse
import curses
import sys
import traceback
from . import __version__
from .menu import Menu

def start():
    version = __version__
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', help='show this version and exit', action='store_true')
    args = parser.parse_args()
    if args.version:
        latest = Menu().check_version()
        try:
            curses.endwin()
        except _curses.error:
            pass
        print('NetEase-MusicBox installed version:' + version)
        if latest != version:
            print('NetEase-MusicBox latest version:' + str(latest))
        sys.exit()
    nembox_menu = Menu()
    try:
        nembox_menu.start_fork(version)
    except (OSError, TypeError, ValueError, KeyError, IndexError):
        try:
            curses.echo()
            curses.nocbreak()
            curses.endwin()
        except _curses.error:
            pass
        traceback.print_exc()
if __name__ == '__main__':
    start()

# File: NEMbox\api.py

"""
网易云音乐 Api
"""
import json
import platform
import time
from collections import OrderedDict
from http.cookiejar import Cookie
from http.cookiejar import MozillaCookieJar
import requests
import requests_cache
from .config import Config
from .const import Constant
from .encrypt import encrypted_request
from .logger import getLogger
from .storage import Storage
requests_cache.install_cache(Constant.cache_path, expire_after=3600)
log = getLogger(__name__)
TOP_LIST_ALL = {0: ['云音乐新歌榜', '3779629'], 1: ['云音乐热歌榜', '3778678'], 2: ['网易原创歌曲榜', '2884035'], 3: ['云音乐飙升榜', '19723756'], 4: ['云音乐电音榜', '10520166'], 5: ['UK排行榜周榜', '180106'], 6: ['美国Billboard周榜', '60198'], 7: ['KTV嗨榜', '21845217'], 8: ['iTunes榜', '11641012'], 9: ['Hit FM Top榜', '120001'], 10: ['日本Oricon周榜', '60131'], 11: ['韩国Melon排行榜周榜', '3733003'], 12: ['韩国Mnet排行榜周榜', '60255'], 13: ['韩国Melon原声周榜', '46772709'], 14: ['中国TOP排行榜(港台榜)', '112504'], 15: ['中国TOP排行榜(内地榜)', '64016'], 16: ['香港电台中文歌曲龙虎榜', '10169002'], 17: ['华语金曲榜', '4395559'], 18: ['中国嘻哈榜', '1899724'], 19: ['法国 NRJ EuroHot 30周榜', '27135204'], 20: ['台湾Hito排行榜', '112463'], 21: ['Beatport全球电子舞曲榜', '3812895'], 22: ['云音乐ACG音乐榜', '71385702'], 23: ['云音乐嘻哈榜', '991319590']}
PLAYLIST_CLASSES = OrderedDict([('语种', ['华语', '欧美', '日语', '韩语', '粤语', '小语种']), ('风格', ['流行', '摇滚', '民谣', '电子', '舞曲', '说唱', '轻音乐', '爵士', '乡村', 'R&B/Soul', '古典', '民族', '英伦', '金属', '朋克', '蓝调', '雷鬼', '世界音乐', '拉丁', '另类/独立', 'New Age', '古风', '后摇', 'Bossa Nova']), ('场景', ['清晨', '夜晚', '学习', '工作', '午休', '下午茶', '地铁', '驾车', '运动', '旅行', '散步', '酒吧']), ('情感', ['怀旧', '清新', '浪漫', '性感', '伤感', '治愈', '放松', '孤独', '感动', '兴奋', '快乐', '安静', '思念']), ('主题', ['影视原声', 'ACG', '儿童', '校园', '游戏', '70后', '80后', '90后', '网络歌曲', 'KTV', '经典', '翻唱', '吉他', '钢琴', '器乐', '榜单', '00后'])])
DEFAULT_TIMEOUT = 10
BASE_URL = 'http://music.163.com'

class Parse(object):

    @classmethod
    def _song_url_by_id(cls, sid):
        url = 'http://music.163.com/song/media/outer/url?id={}.mp3'.format(sid)
        quality = 'LD 128k'
        return (url, quality)

    @classmethod
    def song_url(cls, song):
        if 'url' in song:
            url = song['url']
            if url is None:
                return Parse._song_url_by_id(song['id'])
            br = song['br']
            if br >= 320000:
                quality = 'HD'
            elif br >= 192000:
                quality = 'MD'
            else:
                quality = 'LD'
            return (url, '{} {}k'.format(quality, br // 1000))
        else:
            return Parse._song_url_by_id(song['id'])

    @classmethod
    def song_album(cls, song):
        if 'al' in song:
            if song['al'] is not None:
                album_name = song['al']['name']
                album_id = song['al']['id']
            else:
                album_name = '未知专辑'
                album_id = ''
        elif 'album' in song:
            if song['album'] is not None:
                album_name = song['album']['name']
                album_id = song['album']['id']
            else:
                album_name = '未知专辑'
                album_id = ''
        else:
            raise ValueError
        return (album_name, album_id)

    @classmethod
    def song_artist(cls, song):
        artist = ''
        if 'ar' in song:
            artist = ', '.join([a['name'] for a in song['ar'] if a['name'] is not None])
            if artist == '' and 'pc' in song:
                artist = '未知艺术家' if song['pc']['ar'] is None else song['pc']['ar']
        elif 'artists' in song:
            artist = ', '.join([a['name'] for a in song['artists']])
        else:
            artist = '未知艺术家'
        return artist

    @classmethod
    def songs(cls, songs):
        song_info_list = []
        for song in songs:
            (url, quality) = Parse.song_url(song)
            if not url:
                continue
            (album_name, album_id) = Parse.song_album(song)
            song_info = {'song_id': song['id'], 'artist': Parse.song_artist(song), 'song_name': song['name'], 'album_name': album_name, 'album_id': album_id, 'mp3_url': url, 'quality': quality, 'expires': song['expires'], 'get_time': song['get_time']}
            song_info_list.append(song_info)
        return song_info_list

    @classmethod
    def artists(cls, artists):
        return [{'artist_id': artist['id'], 'artists_name': artist['name'], 'alias': ''.join(artist['alias'])} for artist in artists]

    @classmethod
    def albums(cls, albums):
        return [{'album_id': album['id'], 'albums_name': album['name'], 'artists_name': album['artist']['name']} for album in albums]

    @classmethod
    def playlists(cls, playlists):
        return [{'playlist_id': pl['id'], 'playlist_name': pl['name'], 'creator_name': pl['creator']['nickname']} for pl in playlists]

class NetEase(object):

    def __init__(self):
        self.header = {'Accept': '*/*', 'Accept-Encoding': 'gzip,deflate,sdch', 'Accept-Language': 'zh-CN,zh;q=0.8,gl;q=0.6,zh-TW;q=0.4', 'Connection': 'keep-alive', 'Content-Type': 'application/x-www-form-urlencoded', 'Host': 'music.163.com', 'Referer': 'http://music.163.com', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.87 Safari/537.36'}
        self.storage = Storage()
        cookie_jar = MozillaCookieJar(self.storage.cookie_path)
        cookie_jar.load()
        self.session = requests.Session()
        self.session.cookies = cookie_jar
        for cookie in cookie_jar:
            if cookie.is_expired():
                cookie_jar.clear()
                self.storage.database['user'] = {'username': '', 'password': '', 'user_id': '', 'nickname': ''}
                self.storage.save()
                break

    @property
    def toplists(self):
        return [item[0] for item in TOP_LIST_ALL.values()]

    def logout(self):
        self.session.cookies.clear()
        self.storage.database['user'] = {'username': '', 'password': '', 'user_id': '', 'nickname': ''}
        self.session.cookies.save()
        self.storage.save()

    def _raw_request(self, method, endpoint, data=None):
        resp = None
        if method == 'GET':
            resp = self.session.get(endpoint, params=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        elif method == 'POST':
            resp = self.session.post(endpoint, data=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        return resp

    def make_cookie(self, name, value):
        return Cookie(version=0, name=name, value=value, port=None, port_specified=False, domain='music.163.com', domain_specified=True, domain_initial_dot=False, path='/', path_specified=True, secure=False, expires=None, discard=False, comment=None, comment_url=None, rest={})

    def request(self, method, path, params={}, default={'code': -1}, custom_cookies={}):
        endpoint = '{}{}'.format(BASE_URL, path)
        csrf_token = ''
        for cookie in self.session.cookies:
            if cookie.name == '__csrf':
                csrf_token = cookie.value
                break
        params.update({'csrf_token': csrf_token})
        data = default
        for (key, value) in custom_cookies.items():
            cookie = self.make_cookie(key, value)
            self.session.cookies.set_cookie(cookie)
        params = encrypted_request(params)
        resp = None
        try:
            resp = self._raw_request(method, endpoint, params)
            data = resp.json()
        except requests.exceptions.RequestException as e:
            log.error(e)
        except ValueError:
            log.error('Path: {}, response: {}'.format(path, resp.text[:200]))
        finally:
            return data

    def login(self, username, password):
        self.session.cookies.load()
        if username.isdigit():
            path = '/weapi/login/cellphone'
            params = dict(phone=username, password=password, countrycode='86', rememberLogin='true')
        else:
            path = '/weapi/login'
            params = dict(username=username, password=password, rememberLogin='true')
        data = self.request('POST', path, params, custom_cookies={'os': 'pc'})
        self.session.cookies.save()
        return data

    def daily_task(self, is_mobile=True):
        path = '/weapi/point/dailyTask'
        params = dict(type=0 if is_mobile else 1)
        return self.request('POST', path, params)

    def user_playlist(self, uid, offset=0, limit=50):
        path = '/weapi/user/playlist'
        params = dict(uid=uid, offset=offset, limit=limit)
        return self.request('POST', path, params).get('playlist', [])

    def recommend_resource(self):
        path = '/weapi/v1/discovery/recommend/resource'
        return self.request('POST', path).get('recommend', [])

    def recommend_playlist(self, total=True, offset=0, limit=20):
        path = '/weapi/v1/discovery/recommend/songs'
        params = dict(total=total, offset=offset, limit=limit)
        return self.request('POST', path, params).get('recommend', [])

    def personal_fm(self):
        path = '/weapi/v1/radio/get'
        return self.request('POST', path).get('data', [])

    def fm_like(self, songid, like=True, time=25, alg='itembased'):
        path = '/weapi/radio/like'
        params = dict(alg=alg, trackId=songid, like='true' if like else 'false', time=time)
        return self.request('POST', path, params)['code'] == 200

    def fm_trash(self, songid, time=25, alg='RT'):
        path = '/weapi/radio/trash/add'
        params = dict(songId=songid, alg=alg, time=time)
        return self.request('POST', path, params)['code'] == 200

    def search(self, keywords, stype=1, offset=0, total='true', limit=50):
        path = '/weapi/search/get'
        params = dict(s=keywords, type=stype, offset=offset, total=total, limit=limit)
        return self.request('POST', path, params).get('result', {})

    def new_albums(self, offset=0, limit=50):
        path = '/weapi/album/new'
        params = dict(area='ALL', offset=offset, total=True, limit=limit)
        return self.request('POST', path, params).get('albums', [])

    def top_playlists(self, category='全部', order='hot', offset=0, limit=50):
        path = '/weapi/playlist/list'
        params = dict(cat=category, order=order, offset=offset, total='true', limit=limit)
        return self.request('POST', path, params).get('playlists', [])

    def playlist_catelogs(self):
        path = '/weapi/playlist/catalogue'
        return self.request('POST', path)

    def playlist_songlist(self, playlist_id):
        path = '/weapi/v3/playlist/detail'
        params = dict(id=playlist_id, total='true', limit=1000, n=1000, offest=0)
        custom_cookies = dict(os=platform.system())
        return self.request('POST', path, params, {'code': -1}, custom_cookies).get('playlist', {}).get('trackIds', [])

    def top_artists(self, offset=0, limit=100):
        path = '/weapi/artist/top'
        params = dict(offset=offset, total=True, limit=limit)
        return self.request('POST', path, params).get('artists', [])

    def top_songlist(self, idx=0, offset=0, limit=100):
        playlist_id = TOP_LIST_ALL[idx][1]
        return self.playlist_songlist(playlist_id)

    def artists(self, artist_id):
        path = '/weapi/v1/artist/{}'.format(artist_id)
        return self.request('POST', path).get('hotSongs', [])

    def get_artist_album(self, artist_id, offset=0, limit=50):
        path = '/weapi/artist/albums/{}'.format(artist_id)
        params = dict(offset=offset, total=True, limit=limit)
        return self.request('POST', path, params).get('hotAlbums', [])

    def album(self, album_id):
        path = '/weapi/v1/album/{}'.format(album_id)
        return self.request('POST', path).get('songs', [])

    def song_comments(self, music_id, offset=0, total='false', limit=100):
        path = '/weapi/v1/resource/comments/R_SO_4_{}/'.format(music_id)
        params = dict(rid=music_id, offset=offset, total=total, limit=limit)
        return self.request('POST', path, params)

    def songs_detail(self, ids):
        path = '/weapi/v3/song/detail'
        params = dict(c=json.dumps([{'id': _id} for _id in ids]), ids=json.dumps(ids))
        return self.request('POST', path, params).get('songs', [])

    def songs_url(self, ids):
        quality = Config().get('music_quality')
        rate_map = {0: 320000, 1: 192000, 2: 128000}
        path = '/weapi/song/enhance/player/url'
        params = dict(ids=ids, br=rate_map[quality])
        return self.request('POST', path, params).get('data', [])

    def song_lyric(self, music_id):
        path = '/weapi/song/lyric'
        params = dict(os='osx', id=music_id, lv=-1, kv=-1, tv=-1)
        lyric = self.request('POST', path, params).get('lrc', {}).get('lyric', [])
        if not lyric:
            return []
        else:
            return lyric.split('\n')

    def song_tlyric(self, music_id):
        path = '/weapi/song/lyric'
        params = dict(os='osx', id=music_id, lv=-1, kv=-1, tv=-1)
        lyric = self.request('POST', path, params).get('tlyric', {}).get('lyric', [])
        if not lyric:
            return []
        else:
            return lyric.split('\n')

    def djRadios(self, offset=0, limit=50):
        path = '/weapi/djradio/hot/v1'
        params = dict(limit=limit, offset=offset)
        return self.request('POST', path, params).get('djRadios', [])

    def djprograms(self, radio_id, asc=False, offset=0, limit=50):
        path = '/weapi/dj/program/byradio'
        params = dict(asc=asc, radioId=radio_id, offset=offset, limit=limit)
        programs = self.request('POST', path, params).get('programs', [])
        return [p['mainSong'] for p in programs]

    def alldjprograms(self, radio_id, asc=False, offset=0, limit=500):
        programs = []
        ps = self.djprograms(radio_id, asc=asc, offset=offset, limit=limit)
        while ps:
            programs.extend(ps)
            offset += limit
            ps = self.djprograms(radio_id, asc=asc, offset=offset, limit=limit)
        return programs

    def get_version(self):
        action = 'https://pypi.org/pypi/NetEase-MusicBox/json'
        try:
            return requests.get(action).json()
        except requests.exceptions.RequestException as e:
            log.error(e)
            return {}

    def dig_info(self, data, dig_type):
        if not data:
            return []
        if dig_type == 'songs' or dig_type == 'fmsongs' or dig_type == 'djprograms':
            sids = [x['id'] for x in data]
            urls = []
            for i in range(0, len(sids), 350):
                urls.extend(self.songs_url(sids[i:i + 350]))
            sds = []
            if dig_type == 'djprograms':
                sds.extend(data)
            else:
                for i in range(0, len(sids), 500):
                    sds.extend(self.songs_detail(sids[i:i + 500]))
            url_id_index = {}
            for (index, url) in enumerate(urls):
                url_id_index[url['id']] = index
            timestamp = time.time()
            for s in sds:
                url_index = url_id_index.get(s['id'])
                if url_index is None:
                    log.error("can't get song url, id: %s", s['id'])
                    return []
                s['url'] = urls[url_index]['url']
                s['br'] = urls[url_index]['br']
                s['expires'] = urls[url_index]['expi']
                s['get_time'] = timestamp
            return Parse.songs(sds)
        elif dig_type == 'refresh_urls':
            urls_info = []
            for i in range(0, len(data), 350):
                urls_info.extend(self.songs_url(data[i:i + 350]))
            timestamp = time.time()
            songs = []
            for url_info in urls_info:
                song = {}
                song['song_id'] = url_info['id']
                song['mp3_url'] = url_info['url']
                song['expires'] = url_info['expi']
                song['get_time'] = timestamp
                songs.append(song)
            return songs
        elif dig_type == 'artists':
            return Parse.artists(data)
        elif dig_type == 'albums':
            return Parse.albums(data)
        elif dig_type == 'playlists' or dig_type == 'top_playlists':
            return Parse.playlists(data)
        elif dig_type == 'playlist_classes':
            return list(PLAYLIST_CLASSES.keys())
        elif dig_type == 'playlist_class_detail':
            return PLAYLIST_CLASSES[data]
        elif dig_type == 'djRadios':
            return data
        else:
            raise ValueError('Invalid dig type')

# File: NEMbox\cache.py

"""
Class to cache songs into local storage.
"""
import os
import signal
import subprocess
import threading
from . import logger
from .api import NetEase
from .config import Config
from .const import Constant
from .singleton import Singleton
log = logger.getLogger(__name__)

class Cache(Singleton):

    def __init__(self):
        if hasattr(self, '_init'):
            return
        self._init = True
        self.const = Constant()
        self.config = Config()
        self.download_lock = threading.Lock()
        self.check_lock = threading.Lock()
        self.downloading = []
        self.aria2c = None
        self.wget = None
        self.stop = False
        self.enable = self.config.get('cache')
        self.aria2c_parameters = self.config.get('aria2c_parameters')

    def _is_cache_successful(self):

        def succ(x):
            return x and x.returncode == 0
        return succ(self.aria2c) or succ(self.wget)

    def _kill_all(self):

        def _kill(p):
            if p:
                os.kill(p.pid, signal.SIGKILL)
        _kill(self.aria2c)
        _kill(self.wget)

    def start_download(self):
        check = self.download_lock.acquire(False)
        if not check:
            return False
        while True:
            if self.stop:
                break
            if not self.enable:
                break
            self.check_lock.acquire()
            if len(self.downloading) <= 0:
                self.check_lock.release()
                break
            data = self.downloading.pop()
            self.check_lock.release()
            song_id = data[0]
            song_name = data[1]
            artist = data[2]
            url = data[3]
            onExit = data[4]
            output_path = Constant.download_dir
            output_file = str(artist) + ' - ' + str(song_name) + '.mp3'
            full_path = os.path.join(output_path, output_file)
            new_url = NetEase().songs_url([song_id])[0]['url']
            if new_url:
                log.info('Old:{}. New:{}'.format(url, new_url))
                try:
                    para = ['aria2c', '--auto-file-renaming=false', '--allow-overwrite=true', '-d', output_path, '-o', output_file, new_url]
                    para.extend(self.aria2c_parameters)
                    log.debug(para)
                    self.aria2c = subprocess.Popen(para, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    self.aria2c.wait()
                except OSError as e:
                    log.warning('{}.\tAria2c is unavailable, fall back to wget'.format(e))
                    para = ['wget', '-O', full_path, new_url]
                    self.wget = subprocess.Popen(para, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    self.wget.wait()
                if self._is_cache_successful():
                    log.debug(str(song_id) + ' Cache OK')
                    onExit(song_id, full_path)
        self.download_lock.release()

    def add(self, song_id, song_name, artist, url, onExit):
        self.check_lock.acquire()
        self.downloading.append([song_id, song_name, artist, url, onExit])
        self.check_lock.release()

    def quit(self):
        self.stop = True
        try:
            self._kill_all()
        except (AttributeError, OSError) as e:
            log.error(e)
            pass

# File: NEMbox\cmd_parser.py

"""
捕获类似curses键盘输入流,生成指令流
"""
import curses
from copy import deepcopy
from functools import wraps
from .config import Config
ERASE_SPEED = 5
__all__ = ['cmd_parser', 'parse_keylist', 'coroutine', 'erase_coroutine']
KEY_MAP = Config().get('keymap')

def coroutine(func):

    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return primer

def _cmd_parser():
    """
    A generator receive key value typed by user return constant keylist.
    输入键盘输入流,输出指令流,以curses默认-1为信号终止.
    """
    pre_key = -1
    keylist = []
    while 1:
        key = (yield)
        if key > 0 and pre_key == -1:
            keylist.append(key)
        elif key > 0 and pre_key > 0:
            keylist.append(key)
        elif curses.keyname(key).decode('utf-8') in KEY_MAP.values() and pre_key > 0:
            keylist.append(key)
            return keylist
        pre_key = key

def cmd_parser(results):
    """
    A generator manager which can catch StopIteration and start a new Generator.
    生成器管理对象,可以优雅地屏蔽生成器的终止信号,并重启生成器
    """
    while 1:
        results.clear()
        results += (yield from _cmd_parser())
        yield results

def _erase_coroutine():
    keylist = []
    while 1:
        key = (yield)
        keylist.append(key)
        if len(set(keylist)) > 1:
            return keylist
        elif len(keylist) >= ERASE_SPEED * 2:
            return keylist

def erase_coroutine(erase_cmd_list):
    while 1:
        erase_cmd_list.clear()
        erase_cmd_list += (yield from _erase_coroutine())
        yield erase_cmd_list

def parse_keylist(keylist):
    """
    '2' '3' '4' 'j'  ----> 234 j
    supoort keys  [  ]   j  k  <KEY_UP> <KEY_DOWN>
    """
    keylist = deepcopy(keylist)
    if keylist == []:
        return None
    if set(keylist) | {ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong'])} == {ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong'])}:
        delta_key = keylist.count(ord(KEY_MAP['nextSong'])) - keylist.count(ord(KEY_MAP['prevSong']))
        if delta_key < 0:
            return (-delta_key, ord(KEY_MAP['prevSong']))
        return (delta_key, ord(KEY_MAP['nextSong']))
    tail_cmd = keylist.pop()
    if tail_cmd in range(48, 58) and set(keylist) | set(range(48, 58)) == set(range(48, 58)):
        return int(''.join([chr(i) for i in keylist] + [chr(tail_cmd)]))
    if len(keylist) == 0:
        return (0, tail_cmd)
    if tail_cmd in (ord(KEY_MAP['prevSong']), ord(KEY_MAP['nextSong']), ord(KEY_MAP['down']), ord(KEY_MAP['up'])) and max(keylist) <= 57 and (min(keylist) >= 48):
        return (int(''.join([chr(i) for i in keylist])), tail_cmd)
    return None

def main(data):
    """
    tset code
    测试代码
    """
    results = []
    group = cmd_parser(results)
    next(group)
    for i in data:
        group.send(i)
    group.send(-1)
    print(results)
    next(group)
    for i in data:
        group.send(i)
    group.send(-1)
    print(results)
    x = _cmd_parser()
    print('-----------')
    print(x.send(None))
    print(x.send(1))
    print(x.send(2))
    print(x.send(3))
    print(x.send(3))
    print(x.send(3))
    try:
        print(x.send(-1))
    except Exception as e:
        print(e.value)
if __name__ == '__main__':
    main(list(range(1, 12)[::-1]))

# File: NEMbox\config.py

import json
import os
from .const import Constant
from .singleton import Singleton
from .utils import utf8_data_to_file

class Config(Singleton):

    def __init__(self):
        if hasattr(self, '_init'):
            return
        self._init = True
        self.path = Constant.config_path
        self.default_config = {'version': 8, 'page_length': {'value': 10, 'default': 10, 'describe': 'Entries each page has. Set 0 to adjust automatically.'}, 'cache': {'value': False, 'default': False, 'describe': 'A toggle to enable cache function or not. Set value to true to enable it.'}, 'mpg123_parameters': {'value': [], 'default': [], 'describe': 'The additional parameters when mpg123 start.'}, 'aria2c_parameters': {'value': [], 'default': [], 'describe': 'The additional parameters when aria2c start to download something.'}, 'music_quality': {'value': 0, 'default': 0, 'describe': 'Select the quality of the music. May be useful when network is terrible. 0 for high quality, 1 for medium and 2 for low.'}, 'global_play_pause': {'value': '<ctrl><alt>p', 'default': '<ctrl><alt>p', 'describe': 'Global keybind for play/pause.Uses gtk notation for keybinds.'}, 'global_next': {'value': '<ctrl><alt>j', 'default': '<ctrl><alt>j', 'describe': 'Global keybind for next song.Uses gtk notation for keybinds.'}, 'global_previous': {'value': '<ctrl><alt>k', 'default': '<ctrl><alt>k', 'describe': 'Global keybind for previous song.Uses gtk notation for keybinds.'}, 'notifier': {'value': True, 'default': True, 'describe': 'Notifier when switching songs.'}, 'translation': {'value': True, 'default': True, 'describe': 'Foreign language lyrics translation.'}, 'osdlyrics': {'value': False, 'default': False, 'describe': 'Desktop lyrics for musicbox.'}, 'osdlyrics_transparent': {'value': False, 'default': False, 'describe': 'Desktop lyrics transparent bg.'}, 'osdlyrics_color': {'value': [225, 248, 113], 'default': [225, 248, 113], 'describe': 'Desktop lyrics RGB Color.'}, 'osdlyrics_size': {'value': [600, 60], 'default': [600, 60], 'describe': 'Desktop lyrics area size.'}, 'osdlyrics_font': {'value': ['Decorative', 16], 'default': ['Decorative', 16], 'describe': 'Desktop lyrics font-family and font-size.'}, 'osdlyrics_background': {'value': 'rgba(100, 100, 100, 120)', 'default': 'rgba(100, 100, 100, 120)', 'describe': 'Desktop lyrics background color.'}, 'osdlyrics_on_top': {'value': True, 'default': True, 'describe': 'Desktop lyrics OnTopHint.'}, 'curses_transparency': {'value': False, 'default': False, 'describe': 'Set true to make curses transparency.'}, 'left_margin_ratio': {'value': 5, 'default': 5, 'describe': 'Controls the ratio between width and left margin.Set to 0 to minimize the margin.'}, 'right_margin_ratio': {'value': 5, 'default': 5, 'describe': 'Controls the ratio between width and right margin.Set to 0 to minimize the margin.'}, 'mouse_movement': {'value': False, 'default': False, 'describe': 'Use mouse or touchpad to move.'}, 'input_timeout': {'value': 500, 'default': 500, 'describe': 'The time wait for the next key.'}, 'colors': {'value': {'pair1': [22, 148], 'pair2': [231, 24], 'pair3': [231, 9], 'pair4': [231, 14], 'pair5': [231, 237]}, 'default': {'pair1': [22, 148], 'pair2': [231, 24], 'pair3': [231, 9], 'pair4': [231, 14], 'pair5': [231, 237]}, 'describe': 'xterm-256color theme.'}, 'keymap': {'value': {'down': 'j', 'up': 'k', 'back': 'h', 'forward': 'l', 'prevPage': 'u', 'nextPage': 'd', 'search': 'f', 'prevSong': '[', 'nextSong': ']', 'jumpIndex': 'G', 'playPause': ' ', 'shuffle': '?', 'volume+': '+', 'volume-': '-', 'menu': 'm', 'presentHistory': 'p', 'musicInfo': 'i', 'playingMode': 'P', 'enterAlbum': 'A', 'add': 'a', 'djList': 'z', 'star': 's', 'collection': 'c', 'remove': 'r', 'moveDown': 'J', 'moveUp': 'K', 'like': ',', 'cache': 'C', 'trashFM': '.', 'nextFM': '/', 'quit': 'q', 'quitClear': 'w', 'help': 'y', 'top': 'g', 'bottom': 'G', 'countDown': 't'}, 'describe': 'Keys and function.'}}
        self.config = {}
        if not os.path.isfile(self.path):
            self.generate_config_file()
        with open(self.path, 'r') as config_file:
            try:
                self.config = json.load(config_file)
            except ValueError:
                self.generate_config_file()

    def generate_config_file(self):
        with open(self.path, 'w') as config_file:
            utf8_data_to_file(config_file, json.dumps(self.default_config, indent=2))

    def save_config_file(self):
        with open(self.path, 'w') as config_file:
            utf8_data_to_file(config_file, json.dumps(self.config, indent=2))

    def get(self, name):
        if name not in self.config.keys():
            self.config[name] = self.default_config[name]
            return self.default_config[name]['value']
        return self.config[name]['value']

# File: NEMbox\const.py

import os

class Constant(object):
    if 'XDG_CONFIG_HOME' in os.environ:
        conf_dir = os.path.join(os.environ['XDG_CONFIG_HOME'], 'netease-musicbox')
    else:
        conf_dir = os.path.join(os.path.expanduser('~'), '.netease-musicbox')
    config_path = os.path.join(conf_dir, 'config.json')
    if 'XDG_CACHE_HOME' in os.environ:
        cacheDir = os.path.join(os.environ['XDG_CACHE_HOME'], 'netease-musicbox')
        if not os.path.exists(cacheDir):
            os.mkdir(cacheDir)
        download_dir = os.path.join(cacheDir, 'cached')
        cache_path = os.path.join(cacheDir, 'nemcache')
    else:
        download_dir = os.path.join(conf_dir, 'cached')
        cache_path = os.path.join(conf_dir, 'nemcache')
    if 'XDG_DATA_HOME' in os.environ:
        dataDir = os.path.join(os.environ['XDG_DATA_HOME'], 'netease-musicbox')
        if not os.path.exists(dataDir):
            os.mkdir(dataDir)
        cookie_path = os.path.join(dataDir, 'cookie.txt')
        log_path = os.path.join(dataDir, 'musicbox.log')
        storage_path = os.path.join(dataDir, 'database.json')
    else:
        cookie_path = os.path.join(conf_dir, 'cookie.txt')
        log_path = os.path.join(conf_dir, 'musicbox.log')
        storage_path = os.path.join(conf_dir, 'database.json')

# File: NEMbox\encrypt.py

import base64
import binascii
import hashlib
import json
import os
from Cryptodome.Cipher import AES
__all__ = ['encrypted_id', 'encrypted_request']
MODULUS = '00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b725152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280104e0312ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424d813cfe4875d3e82047b97ddef52741d546b8e289dc6935b3ece0462db0a22b8e7'
PUBKEY = '010001'
NONCE = b'0CoJUm6Qyw8W8jud'

def encrypted_id(id):
    magic = bytearray('3go8&$8*3*3h0k(2)2', 'u8')
    song_id = bytearray(id, 'u8')
    magic_len = len(magic)
    for (i, sid) in enumerate(song_id):
        song_id[i] = sid ^ magic[i % magic_len]
    m = hashlib.md5(song_id)
    result = m.digest()
    result = base64.b64encode(result).replace(b'/', b'_').replace(b'+', b'-')
    return result.decode('utf-8')

def encrypted_request(text):
    data = json.dumps(text).encode('utf-8')
    secret = create_key(16)
    params = aes(aes(data, NONCE), secret)
    encseckey = rsa(secret, PUBKEY, MODULUS)
    return {'params': params, 'encSecKey': encseckey}

def aes(text, key):
    pad = 16 - len(text) % 16
    text = text + bytearray([pad] * pad)
    encryptor = AES.new(key, 2, b'0102030405060708')
    ciphertext = encryptor.encrypt(text)
    return base64.b64encode(ciphertext)

def rsa(text, pubkey, modulus):
    text = text[::-1]
    rs = pow(int(binascii.hexlify(text), 16), int(pubkey, 16), int(modulus, 16))
    return format(rs, 'x').zfill(256)

def create_key(size):
    return binascii.hexlify(os.urandom(size))[:16]

# File: NEMbox\kill_thread.py

import ctypes
import inspect
import threading
import time
__all__ = ['stop_thread']

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError('invalid thread id')
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError('PyThreadState_SetAsyncExc failed')

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)

def test():
    while True:
        print('-------')
        time.sleep(0.5)
if __name__ == '__main__':
    t = threading.Thread(target=test)
    t.start()
    time.sleep(5.2)
    print('main thread sleep finish')
    stop_thread(t)

# File: NEMbox\logger.py

import logging
from . import const
FILE_NAME = const.Constant.log_path
with open(FILE_NAME, 'a+') as f:
    f.write('#' * 80)
    f.write('\n')

def getLogger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(FILE_NAME)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(lineno)s: %(message)s'))
    log.addHandler(fh)
    return log

# File: NEMbox\menu.py

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

def carousel(left, right, x):
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
shortcut = [[KEY_MAP['down'], 'Down', '下移'], [KEY_MAP['up'], 'Up', '上移'], ['<Num>+' + KEY_MAP['up'], '<num> Up', '上移num'], ['<Num>+' + KEY_MAP['down'], '<num> Down', '下移num'], [KEY_MAP['back'], 'Back', '后退'], [KEY_MAP['forward'], 'Forward', '前进'], [KEY_MAP['prevPage'], 'Prev page', '上一页'], [KEY_MAP['nextPage'], 'Next page', '下一页'], [KEY_MAP['search'], 'Search', '快速搜索'], [KEY_MAP['prevSong'], 'Prev song', '上一曲'], [KEY_MAP['nextSong'], 'Next song', '下一曲'], ['<Num>+' + KEY_MAP['nextSong'], '<Num> Next Song', '下num曲'], ['<Num>+' + KEY_MAP['prevSong'], '<Num> Prev song', '上num曲'], ['<Num>', 'Goto song <num>', '跳转指定歌曲id'], [KEY_MAP['playPause'], 'Play/Pause', '播放/暂停'], [KEY_MAP['shuffle'], 'Shuffle', '手气不错'], [KEY_MAP['volume+'], 'Volume+', '音量增加'], [KEY_MAP['volume-'], 'Volume-', '音量减少'], [KEY_MAP['menu'], 'Menu', '主菜单'], [KEY_MAP['presentHistory'], 'Present/History', '当前/历史播放列表'], [KEY_MAP['musicInfo'], 'Music Info', '当前音乐信息'], [KEY_MAP['playingMode'], 'Playing Mode', '播放模式切换'], [KEY_MAP['enterAlbum'], 'Enter album', '进入专辑'], [KEY_MAP['add'], 'Add', '添加曲目到打碟'], [KEY_MAP['djList'], 'DJ list', '打碟列表（退出后清空）'], [KEY_MAP['star'], 'Star', '添加到本地收藏'], [KEY_MAP['collection'], 'Collection', '本地收藏列表'], [KEY_MAP['remove'], 'Remove', '删除当前条目'], [KEY_MAP['moveDown'], 'Move Down', '向下移动当前条目'], [KEY_MAP['moveUp'], 'Move Up', '向上移动当前条目'], [KEY_MAP['like'], 'Like', '喜爱'], [KEY_MAP['cache'], 'Cache', '缓存歌曲到本地'], [KEY_MAP['nextFM'], 'Next FM', '下一 FM'], [KEY_MAP['trashFM'], 'Trash FM', '删除 FM'], [KEY_MAP['quit'], 'Quit', '退出'], [KEY_MAP['quitClear'], 'Quit&Clear', '退出并清除用户信息'], [KEY_MAP['help'], 'Help', '帮助'], [KEY_MAP['top'], 'Top', '回到顶部'], [KEY_MAP['bottom'], 'Bottom', '跳转到底部'], [KEY_MAP['countDown'], 'Count Down', '定时']]

class Menu(object):

    def __init__(self):
        self.quit = False
        self.config = Config()
        self.datatype = 'main'
        self.title = '网易云音乐'
        self.datalist = [{'entry_name': '排行榜'}, {'entry_name': '艺术家'}, {'entry_name': '新碟上架'}, {'entry_name': '精选歌单'}, {'entry_name': '我的歌单'}, {'entry_name': '主播电台'}, {'entry_name': '每日推荐歌曲'}, {'entry_name': '每日推荐歌单'}, {'entry_name': '私人FM'}, {'entry_name': '搜索'}, {'entry_name': '帮助'}]
        self.offset = 0
        self.index = 0
        self.storage = Storage()
        self.storage.load()
        self.collection = self.storage.database['collections']
        self.player = Player()
        self.player.playing_song_changed_callback = self.song_changed_callback
        self.cache = Cache()
        self.ui = Ui()
        self.api = NetEase()
        self.screen = C.initscr()
        self.screen.keypad(1)
        self.step = Config().get('page_length')
        if self.step == 0:
            self.step = max(int(self.ui.y * 4 / 5) - 10, 1)
        self.stack = []
        self.djstack = []
        self.at_playing_list = False
        self.enter_flag = True
        signal.signal(signal.SIGWINCH, self.change_term)
        signal.signal(signal.SIGINT, self.send_kill)
        signal.signal(signal.SIGTERM, self.send_kill)
        self.menu_starts = time.time()
        self.countdown_start = time.time()
        self.countdown = -1
        self.is_in_countdown = False
        self.timer = 0
        self.key_list = []
        self.pre_keylist = []
        self.parser = None
        self.at_search_result = False

    @property
    def user(self):
        return self.storage.database['user']

    @property
    def account(self):
        return self.user['username']

    @property
    def md5pass(self):
        return self.user['password']

    @property
    def userid(self):
        return self.user['user_id']

    @property
    def username(self):
        return self.user['nickname']

    def login(self):
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

    def in_place_search(self):
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

    def search(self, category):
        self.ui.screen.timeout(-1)
        SearchArg = namedtuple('SearchArg', ['prompt', 'api_type', 'post_process'])
        category_map = {'songs': SearchArg('搜索歌曲：', 1, lambda datalist: datalist), 'albums': SearchArg('搜索专辑：', 10, lambda datalist: datalist), 'artists': SearchArg('搜索艺术家：', 100, lambda datalist: datalist), 'playlists': SearchArg('搜索网易精选集：', 1000, lambda datalist: datalist), 'djRadios': SearchArg('搜索主播电台：', 1009, lambda datalist: datalist)}
        (prompt, api_type, post_process) = category_map[category]
        keyword = self.ui.get_param(prompt)
        if not keyword:
            return []
        data = self.api.search(keyword, api_type)
        if not data:
            return data
        datalist = post_process(data.get(category, []))
        return self.api.dig_info(datalist, category)

    def change_term(self, signum, frame):
        self.ui.screen.clear()
        self.ui.screen.refresh()

    def send_kill(self, signum, fram):
        if pyqt_activity:
            stop_lyrics_process()
        self.player.stop()
        self.cache.quit()
        self.storage.save()
        C.endwin()
        sys.exit()

    def update_alert(self, version):
        latest = Menu().check_version()
        if str(latest) > str(version) and latest != 0:
            notify('MusicBox Update == available', 1)
            time.sleep(0.5)
            notify('NetEase-MusicBox installed version:' + version + '\nNetEase-MusicBox latest version:' + latest, 0)

    def check_version(self):
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

    def start_fork(self, version):
        pid = os.fork()
        if pid == 0:
            Menu().update_alert(version)
        else:
            Menu().start()

    def next_song(self):
        if self.player.is_empty:
            return
        self.player.next()

    def previous_song(self):
        if self.player.is_empty:
            return
        self.player.prev()

    def prev_key_event(self):
        self.player.prev_idx()

    def next_key_event(self):
        self.player.next_idx()

    def up_key_event(self):
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

    def down_key_event(self):
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

    def space_key_event_in_search_result(self):
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

    def space_key_event(self):
        idx = self.index
        datatype = self.datatype
        if not self.datalist:
            return
        if idx < 0 or idx >= len(self.datalist):
            self.player.info['idx'] = 0
        datatype_callback = {'songs': None, 'djprograms': None, 'fmsongs': self.fm_callback}
        if datatype in ['songs', 'djprograms', 'fmsongs']:
            self.player.new_player_list(datatype, self.title, self.datalist, -1)
            self.player.end_callback = datatype_callback[datatype]
            self.player.play_or_pause(idx, self.at_playing_list)
            self.at_playing_list = True
        else:
            is_not_songs = True
            self.player.play_or_pause(self.player.info['idx'], is_not_songs)
        self.build_menu_processbar()

    def like_event(self):
        return_data = self.request_api(self.api.fm_like, self.player.playing_id)
        if return_data:
            song_name = self.player.playing_name
            notify('%s added successfully!' % song_name, 0)
        else:
            notify('Adding song failed!', 0)

    def back_page_event(self):
        if len(self.stack) == 1:
            return
        self.menu_starts = time.time()
        (self.datatype, self.title, self.datalist, self.offset, self.index) = self.stack.pop()
        self.at_playing_list = False
        self.at_search_result = False

    def enter_page_event(self):
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

    def album_key_event(self):
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

    def num_jump_key_event(self):
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

    def digit_key_song_event(self):
        """直接跳到指定id 歌曲"""
        step = self.step
        self.key_list.pop()
        song_index = parse_keylist(self.key_list)
        if self.index != song_index:
            self.index = song_index
            self.offset = self.index - self.index % step
            self.build_menu_processbar()
            self.ui.screen.refresh()

    def time_key_event(self):
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

    def down_page_event(self):
        offset = self.offset
        datalist = self.datalist
        step = self.step
        if offset + step >= len(datalist):
            return
        self.menu_starts = time.time()
        self.offset += step
        self.index = (self.index + step) // step * step

    def up_page_event(self):
        offset = self.offset
        step = self.step
        if offset == 0:
            return
        self.menu_starts = time.time()
        self.offset -= step
        self.index = (self.index - step) // step * step

    def resize_key_event(self):
        self.player.update_size()

    def build_menu_processbar(self):
        self.ui.build_process_bar(self.player.current_song, self.player.process_location, self.player.process_length, self.player.playing_flag, self.player.info['playing_mode'])
        self.ui.build_menu(self.datatype, self.title, self.datalist, self.offset, self.index, self.step, self.menu_starts)

    def quit_event(self):
        self.config.save_config_file()
        sys.exit(0)

    def stop(self):
        self.quit = True
        self.player.stop()
        self.cache.quit()
        self.storage.save()
        C.endwin()

    def start(self):
        self.menu_starts = time.time()
        self.ui.build_menu(self.datatype, self.title, self.datalist, self.offset, self.index, self.step, self.menu_starts)
        self.stack.append([self.datatype, self.title, self.datalist, self.offset, self.index])
        if pyqt_activity:
            show_lyrics_new_process()
        pre_key = -1
        keylist = self.key_list
        self.parser = cmd_parser(keylist)
        erase_cmd_list = []
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

    def dispatch_enter(self, idx):
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

    def show_playing_song(self):
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

    def song_changed_callback(self):
        if self.at_playing_list:
            self.show_playing_song()

    def fm_callback(self):
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

    def request_api(self, func, *args):
        result = func(*args)
        if result:
            return result
        if not self.login():
            notify('You need to log in')
            return False
        return func(*args)

    def get_new_fm(self):
        data = self.request_api(self.api.personal_fm)
        if not data:
            return []
        return self.api.dig_info(data, 'fmsongs')

    def choice_channel(self, idx):
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

# File: NEMbox\osdlyrics.py

import sys
from multiprocessing import Process
from multiprocessing import set_start_method
from . import logger
from .config import Config
log = logger.getLogger(__name__)
config = Config()
try:
    from qtpy import QtGui, QtCore, QtWidgets
    import dbus
    import dbus.service
    import dbus.mainloop.glib
    pyqt_activity = True
except ImportError:
    pyqt_activity = False
    log.warn('qtpy module not installed.')
    log.warn('Osdlyrics Not Available.')
if pyqt_activity:
    QWidget = QtWidgets.QWidget
    QApplication = QtWidgets.QApplication

    class Lyrics(QWidget):

        def __init__(self):
            super(Lyrics, self).__init__()
            self.text = ''
            self.initUI()

        def initUI(self):
            self.setStyleSheet('background:' + config.get('osdlyrics_background'))
            if config.get('osdlyrics_transparent'):
                self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
            self.setAttribute(QtCore.Qt.WA_X11DoNotAcceptFocus)
            self.setFocusPolicy(QtCore.Qt.NoFocus)
            if config.get('osdlyrics_on_top'):
                self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.X11BypassWindowManagerHint)
            else:
                self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            self.setMinimumSize(600, 50)
            osdlyrics_size = config.get('osdlyrics_size')
            self.resize(osdlyrics_size[0], osdlyrics_size[1])
            scn = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
            bl = QApplication.desktop().screenGeometry(scn).bottomLeft()
            br = QApplication.desktop().screenGeometry(scn).bottomRight()
            bc = (bl + br) / 2
            frameGeo = self.frameGeometry()
            frameGeo.moveCenter(bc)
            frameGeo.moveBottom(bc.y())
            self.move(frameGeo.topLeft())
            self.text = 'OSD Lyrics for Musicbox'
            self.setWindowTitle('Lyrics')
            self.show()

        def mousePressEvent(self, event):
            self.mpos = event.pos()

        def mouseMoveEvent(self, event):
            if event.buttons() and QtCore.Qt.LeftButton:
                diff = event.pos() - self.mpos
                newpos = self.pos() + diff
                self.move(newpos)

        def wheelEvent(self, event):
            self.resize(self.width() + event.delta(), self.height())

        def paintEvent(self, event):
            qp = QtGui.QPainter()
            qp.begin(self)
            self.drawText(event, qp)
            qp.end()

        def drawText(self, event, qp):
            osdlyrics_color = config.get('osdlyrics_color')
            osdlyrics_font = config.get('osdlyrics_font')
            font = QtGui.QFont(osdlyrics_font[0], osdlyrics_font[1])
            pen = QtGui.QColor(osdlyrics_color[0], osdlyrics_color[1], osdlyrics_color[2])
            qp.setFont(font)
            qp.setPen(pen)
            qp.drawText(event.rect(), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, self.text)

        def setText(self, text):
            self.text = text
            self.repaint()

    class LyricsAdapter(dbus.service.Object):

        def __init__(self, name, session):
            dbus.service.Object.__init__(self, name, session)
            self.widget = Lyrics()

        @dbus.service.method('local.musicbox.Lyrics', in_signature='s', out_signature='')
        def refresh_lyrics(self, text):
            self.widget.setText(text.replace('||', '\n'))

        @dbus.service.method('local.musicbox.Lyrics', in_signature='', out_signature='')
        def exit(self):
            QApplication.quit()

    def show_lyrics():
        app = QApplication(sys.argv)
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        session_bus = dbus.SessionBus()
        name = dbus.service.BusName('org.musicbox.Bus', session_bus)
        lyrics = LyricsAdapter(session_bus, '/')
        app.exec_()

def stop_lyrics_process():
    if pyqt_activity:
        bus = dbus.SessionBus().get_object('org.musicbox.Bus', '/')
        bus.exit(dbus_interface='local.musicbox.Lyrics')

def show_lyrics_new_process():
    if pyqt_activity and config.get('osdlyrics'):
        set_start_method('spawn')
        p = Process(target=show_lyrics)
        p.daemon = True
        p.start()

# File: NEMbox\player.py

"""
网易云音乐 Player
"""
import os
import random
import subprocess
import threading
import time
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
    MODE_ORDERED = 0
    MODE_ORDERED_LOOP = 1
    MODE_SINGLE_LOOP = 2
    MODE_RANDOM = 3
    MODE_RANDOM_LOOP = 4
    SUBPROCESS_LIST = []
    MUSIC_THREADS = []

    def __init__(self):
        self.config = Config()
        self.ui = Ui()
        self.popen_handler = None
        self.playing_flag = False
        self.refresh_url_flag = False
        self.process_length = 0
        self.process_location = 0
        self.storage = Storage()
        self.cache = Cache()
        self.end_callback = None
        self.playing_song_changed_callback = None
        self.api = NetEase()
        self.playinfo_starts = time.time()

    @property
    def info(self):
        return self.storage.database['player_info']

    @property
    def songs(self):
        return self.storage.database['songs']

    @property
    def index(self):
        return self.info['idx']

    @property
    def list(self):
        return self.info['player_list']

    @property
    def order(self):
        return self.info['playing_order']

    @property
    def mode(self):
        return self.info['playing_mode']

    @property
    def is_ordered_mode(self):
        return self.mode == Player.MODE_ORDERED

    @property
    def is_ordered_loop_mode(self):
        return self.mode == Player.MODE_ORDERED_LOOP

    @property
    def is_single_loop_mode(self):
        return self.mode == Player.MODE_SINGLE_LOOP

    @property
    def is_random_mode(self):
        return self.mode == Player.MODE_RANDOM

    @property
    def is_random_loop_mode(self):
        return self.mode == Player.MODE_RANDOM_LOOP

    @property
    def config_notifier(self):
        return self.config.get('notifier')

    @property
    def config_mpg123(self):
        return self.config.get('mpg123_parameters')

    @property
    def current_song(self):
        if not self.songs:
            return {}
        if not self.is_index_valid:
            return {}
        song_id = self.list[self.index]
        return self.songs.get(song_id, {})

    @property
    def playing_id(self):
        return self.current_song.get('song_id')

    @property
    def playing_name(self):
        return self.current_song.get('song_name')

    @property
    def is_empty(self):
        return len(self.list) == 0

    @property
    def is_index_valid(self):
        return 0 <= self.index < len(self.list)

    def notify_playing(self):
        if not self.current_song:
            return
        if not self.config_notifier:
            return
        song = self.current_song
        notify('正在播放: {}\n{}-{}'.format(song['song_name'], song['artist'], song['album_name']))

    def notify_copyright_issue(self):
        log.warning('Song {} is unavailable due to copyright issue.'.format(self.playing_id))
        notify('版权限制，无法播放此歌曲')

    def change_mode(self, step=1):
        self.info['playing_mode'] = (self.info['playing_mode'] + step) % 5

    def build_playinfo(self):
        if not self.current_song:
            return
        self.ui.build_playinfo(self.current_song['song_name'], self.current_song['artist'], self.current_song['album_name'], self.current_song['quality'], self.playinfo_starts, pause=not self.playing_flag)

    def add_songs(self, songs):
        for song in songs:
            song_id = str(song['song_id'])
            self.info['player_list'].append(song_id)
            if song_id in self.songs:
                self.songs[song_id].update(song)
            else:
                self.songs[song_id] = song

    def refresh_urls(self):
        songs = self.api.dig_info(self.list, 'refresh_urls')
        if songs:
            for song in songs:
                song_id = str(song['song_id'])
                if song_id in self.songs:
                    self.songs[song_id]['mp3_url'] = song['mp3_url']
                    self.songs[song_id]['expires'] = song['expires']
                    self.songs[song_id]['get_time'] = song['get_time']
                else:
                    self.songs[song_id] = song
            self.refresh_url_flag = True

    def stop(self):
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
                        pass

    def tune_volume(self, up=0):
        try:
            if self.popen_handler.poll():
                return
        except Exception as e:
            log.warn('Unable to tune volume: ' + str(e))
            return
        new_volume = self.info['playing_volume'] + up
        if new_volume < 0:
            new_volume = 0
        self.info['playing_volume'] = new_volume
        try:
            self.popen_handler.stdin.write('V {}\n'.format(self.info['playing_volume']).encode())
            self.popen_handler.stdin.flush()
        except Exception as e:
            log.warn(e)

    def switch(self):
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

    def run_mpg123(self, on_exit, url, expires=-1, get_time=-1):
        para = ['mpg123', '-R'] + self.config_mpg123
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
        except:
            pass
        strout = ' '
        copyright_issue_flag = False
        frame_cnt = 0
        while True:
            if not hasattr(self.popen_handler, 'poll') or self.popen_handler.poll():
                break
            if self.popen_handler.stdout.closed:
                break
            try:
                stroutlines = self.popen_handler.stdout.readline()
            except Exception as e:
                log.warn(e)
                break
            if not stroutlines:
                strout = ' '
                break
            else:
                strout_new = stroutlines.decode().strip()
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
            self.start_playing(lambda : 0, self.current_song)
            self.refresh_url_flag = False
        elif not self.playing_flag:
            self.stop()
        elif copyright_issue_flag and self.is_single_loop_mode:
            self.stop()
        else:
            self.next()

    def download_lyric(self, is_transalted=False):
        key = 'lyric' if not is_transalted else 'tlyric'
        if key not in self.songs[str(self.playing_id)]:
            self.songs[str(self.playing_id)][key] = []
        if len(self.songs[str(self.playing_id)][key]) > 0:
            return
        if not is_transalted:
            lyric = self.api.song_lyric(self.playing_id)
        else:
            lyric = self.api.song_tlyric(self.playing_id)
        self.songs[str(self.playing_id)][key] = lyric

    def download_song(self, song_id, song_name, artist, url):

        def write_path(song_id, path):
            self.songs[str(song_id)]['cache'] = path
        self.cache.add(song_id, song_name, artist, url, write_path)
        self.cache.start_download()

    def start_playing(self, on_exit, args):
        """
        Runs the given args in subprocess.Popen, and then calls the function
        on_exit when the subprocess completes.
        on_exit is a callable object, and args is a lists/tuple of args
        that would give to subprocess.Popen.
        """
        if 'cache' in args.keys() and os.path.isfile(args['cache']):
            thread = threading.Thread(target=self.run_mpg123, args=(on_exit, args['cache']))
        else:
            thread = threading.Thread(target=self.run_mpg123, args=(on_exit, args['mp3_url'], args['expires'], args['get_time']))
            cache_thread = threading.Thread(target=self.download_song, args=(args['song_id'], args['song_name'], args['artist'], args['mp3_url']))
            cache_thread.start()
        thread.start()
        self.MUSIC_THREADS.append(thread)
        self.MUSIC_THREADS = [i for i in self.MUSIC_THREADS if i.is_alive()]
        lyric_download_thread = threading.Thread(target=self.download_lyric)
        lyric_download_thread.start()
        tlyric_download_thread = threading.Thread(target=self.download_lyric, args=(True,))
        tlyric_download_thread.start()
        return thread

    def replay(self):
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
        self.start_playing(lambda : 0, self.current_song)

    def shuffle_order(self):
        del self.order[:]
        self.order.extend(list(range(0, len(self.list))))
        random.shuffle(self.order)
        self.info['random_index'] = 0

    def new_player_list(self, type, title, datalist, offset):
        self.info['player_list_type'] = type
        self.info['player_list_title'] = title
        self.info['player_list'] = []
        self.info['playing_order'] = []
        self.info['random_index'] = 0
        self.add_songs(datalist)

    def append_songs(self, datalist):
        self.add_songs(datalist)

    def play_or_pause(self, idx, switch_flag):
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

    def _swap_song(self):
        now_songs = self.order.index(self.index)
        (self.order[0], self.order[now_songs]) = (self.order[now_songs], self.order[0])

    def _need_to_shuffle(self):
        playing_order = self.order
        random_index = self.info['random_index']
        if random_index >= len(playing_order) or playing_order[random_index] != self.index:
            return True
        else:
            return False

    def next_idx(self):
        if not self.is_index_valid:
            return self.stop()
        playlist_len = len(self.list)
        if self.mode == Player.MODE_ORDERED:
            if self.info['idx'] < playlist_len:
                self.info['idx'] += 1
        elif self.mode == Player.MODE_ORDERED_LOOP:
            self.info['idx'] = (self.index + 1) % playlist_len
        elif self.mode == Player.MODE_SINGLE_LOOP:
            self.info['idx'] = self.info['idx']
        else:
            playing_order_len = len(self.order)
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

    def next(self):
        self.stop()
        self.next_idx()
        self.replay()

    def prev_idx(self):
        if not self.is_index_valid:
            self.stop()
            return
        playlist_len = len(self.list)
        if self.mode == Player.MODE_ORDERED:
            if self.info['idx'] > 0:
                self.info['idx'] -= 1
        elif self.mode == Player.MODE_ORDERED_LOOP:
            self.info['idx'] = (self.info['idx'] - 1) % playlist_len
        elif self.mode == Player.MODE_SINGLE_LOOP:
            self.info['idx'] = self.info['idx']
        else:
            playing_order_len = len(self.order)
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

    def prev(self):
        self.stop()
        self.prev_idx()
        self.replay()

    def shuffle(self):
        self.stop()
        self.info['playing_mode'] = Player.MODE_RANDOM
        self.shuffle_order()
        self.info['idx'] = self.info['playing_order'][self.info['random_index']]
        self.replay()

    def volume_up(self):
        self.tune_volume(5)

    def volume_down(self):
        self.tune_volume(-5)

    def update_size(self):
        self.ui.update_size()
        self.build_playinfo()

    def cache_song(self, song_id, song_name, artist, song_url):

        def on_exit(song_id, path):
            self.songs[str(song_id)]['cache'] = path
            self.cache.enable = False
        self.cache.enable = True
        self.cache.add(song_id, song_name, artist, song_url, on_exit)
        self.cache.start_download()

# File: NEMbox\scrollstring.py

from time import time

class scrollstring(object):

    def __init__(self, content, START):
        self.content = content
        self.display = content
        self.START = START // 1
        self.update()

    def update(self):
        self.display = self.content
        curTime = time() // 1
        offset = max(int((curTime - self.START) % len(self.content)) - 1, 0)
        while offset > 0:
            if self.display[0] > chr(127):
                offset -= 1
                self.display = self.display[1:] + self.display[:1]
            else:
                offset -= 1
                self.display = self.display[2:] + self.display[:2]

    def __repr__(self):
        return self.display

def truelen(string):
    """
    It appears one Asian character takes two spots, but __len__
    counts it as three, so this function counts the dispalyed
    length of the string.

    >>> truelen('abc')
    3
    >>> truelen('你好')
    4
    >>> truelen('1二3')
    4
    >>> truelen('')
    0
    """
    return len(string) + sum((1 for c in string if c > chr(127)))

def truelen_cut(string, length):
    current_length = 0
    current_pos = 0
    for c in string:
        current_length += 2 if c > chr(127) else 1
        if current_length > length:
            return string[:current_pos]
        current_pos += 1
    return string

# File: NEMbox\singleton.py

class Singleton(object):
    """Singleton Class
    This is a class to make some class being a Singleton class.
    Such as database class or config class.

    usage:
        class xxx(Singleton):
            def __init__(self):
                if hasattr(self, '_init'):
                    return
                self._init = True
                other init method
    """

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

# File: NEMbox\storage.py

"""
Class to stores everything into a json file.
"""
import json
from .const import Constant
from .singleton import Singleton
from .utils import utf8_data_to_file

class Storage(Singleton):

    def __init__(self):
        """
        Database stores every info.

        version int
        # if value in file is unequal to value defined in this class.
        # An database update will be applied.
        user dict:
            username str
            key str
        collections list:
            collection_info(dict):
                collection_name str
                collection_type str
                collection_describe str
                collection_songs list:
                    song_id(int)
        songs dict:
            song_id(int) dict:
                song_id int
                artist str
                song_name str
                mp3_url str
                album_name str
                album_id str
                quality str
                lyric str
                tlyric str
        player_info dict:
            player_list list[dict]
            playing_order list[int]
            playing_mode int
            playing_offset int


        :return:
        """
        if hasattr(self, '_init'):
            return
        self._init = True
        self.database = {'user': {'username': '', 'password': '', 'user_id': '', 'nickname': ''}, 'collections': [], 'songs': {}, 'player_info': {'player_list': [], 'player_list_type': '', 'player_list_title': '', 'playing_order': [], 'playing_mode': 0, 'idx': 0, 'ridx': 0, 'playing_volume': 60}}
        self.storage_path = Constant.storage_path
        self.cookie_path = Constant.cookie_path

    def login(self, username, password, userid, nickname):
        self.database['user'] = dict(username=username, password=password, user_id=userid, nickname=nickname)

    def logout(self):
        self.database['user'] = {'username': '', 'password': '', 'user_id': '', 'nickname': ''}

    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                for (k, v) in json.load(f).items():
                    if isinstance(self.database[k], dict):
                        self.database[k].update(v)
                    else:
                        self.database[k] = v
        except (OSError, KeyError, ValueError):
            pass
        self.save()

    def save(self):
        with open(self.storage_path, 'w') as f:
            data = json.dumps(self.database)
            utf8_data_to_file(f, data)

# File: NEMbox\ui.py

"""
网易云音乐 Ui
"""
import curses
import datetime
import os
import re
from shutil import get_terminal_size
from . import logger
from .config import Config
from .scrollstring import scrollstring
from .scrollstring import truelen
from .scrollstring import truelen_cut
from .storage import Storage
from .utils import md5
log = logger.getLogger(__name__)
try:
    import dbus
    dbus_activity = True
except ImportError:
    dbus_activity = False
    log.warn('Qt dbus module is not installed.')
    log.warn('Osdlyrics is not available.')

def break_substr(s, start, max_len=80):
    if truelen(s) <= max_len:
        return s
    res = []
    current_truelen = 0
    start_pos = 0
    end_pos = 0
    for c in s:
        current_truelen += 2 if c > chr(127) else 1
        if current_truelen > max_len:
            res.append(s[start_pos:end_pos])
            current_truelen = 0
            start_pos = end_pos + 1
            end_pos += 1
        else:
            end_pos += 1
    try:
        res.append(s[start_pos:end_pos])
    except Exception:
        pass
    return '\n{}'.format(' ' * start).join(res)

def break_str(s, start, max_len=80):
    res = []
    for substr in s.splitlines():
        res.append(break_substr(substr, start, max_len))
    return '\n{}'.format(' ' * start).join(res)

class Ui(object):

    def __init__(self):
        self.screen = curses.initscr()
        curses.start_color()
        if Config().get('curses_transparency'):
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_CYAN, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
        else:
            colors = Config().get('colors')
            if 'TERM' in os.environ and os.environ['TERM'] == 'xterm-256color' and colors:
                curses.use_default_colors()
                for i in range(1, 6):
                    color = colors['pair' + str(i)]
                    curses.init_pair(i, color[0], color[1])
                self.screen.bkgd(32, curses.color_pair(5))
            else:
                curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
                curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
                curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
                curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        self.config = Config()
        size = get_terminal_size()
        self.x = size[0]
        self.y = size[1]
        self.playerX = 1
        self.playerY = 0
        self.update_margin()
        self.update_space()
        self.lyric = ''
        self.now_lyric = ''
        self.post_lyric = ''
        self.now_lyric_index = 0
        self.now_tlyric_index = 0
        self.tlyric = ''
        self.storage = Storage()
        self.newversion = False

    def addstr(self, *args):
        if len(args) == 1:
            self.screen.addstr(args[0])
        else:
            try:
                self.screen.addstr(args[0], args[1], args[2].encode('utf-8'), *args[3:])
            except Exception as e:
                log.error(e)

    def update_margin(self):
        self.left_margin_ratio = self.config.get('left_margin_ratio')
        if self.left_margin_ratio == 0:
            self.startcol = 0
        else:
            self.startcol = max(int(float(self.x) / self.left_margin_ratio), 0)
        self.indented_startcol = max(self.startcol - 3, 0)
        self.right_margin_ratio = self.config.get('right_margin_ratio')
        if self.right_margin_ratio == 0:
            self.endcol = 0
        else:
            self.endcol = max(int(float(self.x) - float(self.x) / self.right_margin_ratio), self.startcol + 1)
        self.indented_endcol = max(self.endcol - 3, 0)
        self.content_width = self.endcol - self.startcol - 1

    def build_playinfo(self, song_name, artist, album_name, quality, start, pause=False):
        curses.noecho()
        curses.curs_set(0)
        self.screen.move(1, 1)
        self.screen.clrtoeol()
        self.screen.move(2, 1)
        self.screen.clrtoeol()
        if pause:
            self.addstr(1, self.indented_startcol, '_ _ z Z Z ' + quality, curses.color_pair(3))
        else:
            self.addstr(1, self.indented_startcol, '♫  ♪ ♫  ♪ ' + quality, curses.color_pair(3))
        if artist or album_name:
            song_info = '{}{}{}  < {} >'.format(song_name, self.space, artist, album_name)
        else:
            song_info = song_name
        if truelen(song_info) <= self.endcol - self.indented_startcol - 19:
            self.addstr(1, min(self.indented_startcol + 18, self.indented_endcol - 1), song_info, curses.color_pair(4))
        else:
            song_info = scrollstring(song_info + ' ', start)
            self.addstr(1, min(self.indented_startcol + 18, self.indented_endcol - 1), truelen_cut(str(song_info), self.endcol - self.indented_startcol - 19), curses.color_pair(4))
        self.screen.refresh()

    def update_lyrics(self, now_playing, lyrics, tlyrics):
        timestap_regex = '[0-5][0-9]:[0-5][0-9]\\.[0-9]*'

        def get_timestap(lyric_line):
            match_ret = re.match('\\[(' + timestap_regex + ')\\]', lyric_line)
            if match_ret:
                return match_ret.group(1)
            else:
                return ''

        def get_lyric_time(lyric_line):
            lyric_timestap = get_timestap(lyric_line)
            if lyric_timestap == '':
                return datetime.timedelta(seconds=now_playing)
            else:
                return datetime.datetime.strptime(get_timestap(lyric_line), '%M:%S.%f') - datetime.datetime.strptime('00:00', '%M:%S') - lyric_time_offset

        def strip_timestap(lyric_line):
            return re.sub('\\[' + timestap_regex + '\\]', '', lyric_line)

        def append_translation(translated_lyric, origin_lyric):
            translated_lyric = strip_timestap(translated_lyric)
            origin_lyric = strip_timestap(origin_lyric)
            if translated_lyric == '' or origin_lyric == '':
                return translated_lyric + origin_lyric
            return translated_lyric + ' || ' + origin_lyric
        if tlyrics and self.now_tlyric_index >= len(tlyrics) - 1 or self.now_lyric_index >= len(lyrics) - 1:
            self.post_lyric = ''
            return
        lyric_time_offset = datetime.timedelta(seconds=0.5)
        next_lyric_time = get_lyric_time(lyrics[self.now_lyric_index + 1])
        now_time = datetime.timedelta(seconds=now_playing)
        while now_time >= next_lyric_time and self.now_lyric_index < len(lyrics) - 2:
            self.now_lyric_index = self.now_lyric_index + 1
            next_lyric_time = get_lyric_time(lyrics[self.now_lyric_index + 1])
        if tlyrics:
            next_tlyric_time = get_lyric_time(tlyrics[self.now_tlyric_index + 1])
            while now_time >= next_tlyric_time and self.now_tlyric_index < len(tlyrics) - 2:
                self.now_tlyric_index = self.now_tlyric_index + 1
                next_tlyric_time = get_lyric_time(tlyrics[self.now_tlyric_index + 1])
        if tlyrics:
            self.now_lyric = append_translation(tlyrics[self.now_tlyric_index], lyrics[self.now_lyric_index])
            if self.now_tlyric_index < len(tlyrics) - 1 and self.now_lyric_index < len(lyrics) - 1:
                self.post_lyric = append_translation(tlyrics[self.now_tlyric_index + 1], lyrics[self.now_lyric_index + 1])
            else:
                self.post_lyric = ''
        else:
            self.now_lyric = strip_timestap(lyrics[self.now_lyric_index])
            if self.now_lyric_index < len(lyrics) - 1:
                self.post_lyric = strip_timestap(lyrics[self.now_lyric_index + 1])
            else:
                self.post_lyric = ''

    def build_process_bar(self, song, now_playing, total_length, playing_flag, playing_mode):
        if not song or not playing_flag:
            return
        (name, artist) = (song['song_name'], song['artist'])
        (lyrics, tlyrics) = (song.get('lyric', []), song.get('tlyric', []))
        curses.noecho()
        curses.curs_set(0)
        self.screen.move(3, 1)
        self.screen.clrtoeol()
        self.screen.move(4, 1)
        self.screen.clrtoeol()
        self.screen.move(5, 1)
        self.screen.clrtoeol()
        self.screen.move(6, 1)
        self.screen.clrtoeol()
        if total_length <= 0:
            total_length = 1
        if now_playing > total_length or now_playing <= 0:
            now_playing = 0
        if int(now_playing) == 0:
            self.now_lyric_index = 0
            if tlyrics:
                self.now_tlyric_index = 0
            self.now_lyric = ''
            self.post_lyric = ''
        process = '['
        process_bar_width = self.content_width - 24
        for i in range(0, process_bar_width):
            if i < now_playing / total_length * process_bar_width:
                if i + 1 > now_playing / total_length * process_bar_width:
                    if playing_flag:
                        process += '>'
                        continue
                process += '='
            else:
                process += ' '
        process += '] '
        now = str(datetime.timedelta(seconds=int(now_playing))).lstrip('0').lstrip(':')
        total = str(datetime.timedelta(seconds=total_length)).lstrip('0').lstrip(':')
        process += '({}/{})'.format(now, total)
        if playing_mode == 0:
            process = '顺序播放 ' + process
        elif playing_mode == 1:
            process = '顺序循环 ' + process
        elif playing_mode == 2:
            process = '单曲循环 ' + process
        elif playing_mode == 3:
            process = '随机播放 ' + process
        elif playing_mode == 4:
            process = '随机循环 ' + process
        else:
            pass
        self.addstr(3, self.startcol - 2, process, curses.color_pair(1))
        if not lyrics:
            self.now_lyric = '暂无歌词 ~>_<~ \n'
            self.post_lyric = ''
            if dbus_activity and self.config.get('osdlyrics'):
                self.now_playing = '{} - {}\n'.format(name, artist)
        else:
            self.update_lyrics(now_playing, lyrics, tlyrics)
        if dbus_activity and self.config.get('osdlyrics'):
            try:
                bus = dbus.SessionBus().get_object('org.musicbox.Bus', '/')
                if self.now_lyric == '暂无歌词 ~>_<~ \n':
                    bus.refresh_lyrics(self.now_playing, dbus_interface='local.musicbox.Lyrics')
                else:
                    bus.refresh_lyrics(self.now_lyric, dbus_interface='local.musicbox.Lyrics')
            except Exception as e:
                log.error(e)
                pass
        if self.now_lyric_index % 2 == 0:
            self.addstr(4, max(self.startcol - 2, 0), str(self.now_lyric), curses.color_pair(3))
            self.addstr(5, max(self.startcol + 1, 0), str(self.post_lyric), curses.A_DIM)
        else:
            self.addstr(4, max(self.startcol - 2, 0), str(self.post_lyric), curses.A_DIM)
            self.addstr(5, max(self.startcol + 1, 0), str(self.now_lyric), curses.color_pair(3))
        self.screen.refresh()

    def build_loading(self):
        curses.curs_set(0)
        self.addstr(7, self.startcol, '享受高品质音乐，loading...', curses.color_pair(1))
        self.screen.refresh()

    def build_submenu(self, data):
        pass

    def build_menu(self, datatype, title, datalist, offset, index, step, start):
        curses.noecho()
        curses.curs_set(0)
        self.screen.move(7, 1)
        self.screen.clrtobot()
        self.addstr(7, self.startcol, title, curses.color_pair(1))
        if len(datalist) == 0:
            self.addstr(8, self.startcol, '这里什么都没有 -，-')
            return self.screen.refresh()
        if datatype == 'main':
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i]['entry_name'], curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, str(i) + '. ' + datalist[i]['entry_name'])
        elif datatype == 'songs' or datatype == 'djprograms' or datatype == 'fmsongs':
            iter_range = min(len(datalist), offset + step)
            for i in range(offset, iter_range):
                if isinstance(datalist[i], str):
                    raise ValueError(datalist)
                if i == index:
                    self.addstr(i - offset + 9, 0, ' ' * self.startcol)
                    lead = '-> ' + str(i) + '. '
                    self.addstr(i - offset + 9, self.indented_startcol, lead, curses.color_pair(2))
                    name = '{}{}{}  < {} >'.format(datalist[i]['song_name'], self.space, datalist[i]['artist'], datalist[i]['album_name'])
                    if truelen(name) < self.content_width:
                        self.addstr(i - offset + 9, self.indented_startcol + len(lead), name, curses.color_pair(2))
                    else:
                        name = scrollstring(name + '  ', start)
                        self.addstr(i - offset + 9, self.indented_startcol + len(lead), truelen_cut(str(name), self.content_width - len(str(i)) - 2), curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, 0, ' ' * self.startcol)
                    self.addstr(i - offset + 9, self.startcol, truelen_cut('{}. {}{}{}  < {} >'.format(i, datalist[i]['song_name'], self.space, datalist[i]['artist'], datalist[i]['album_name']), self.content_width))
            self.addstr(iter_range - offset + 9, 0, ' ' * self.x)
        elif datatype == 'comments':
            for i in range(offset, min(len(datalist), offset + step)):
                maxlength = min(self.content_width, truelen(datalist[i]['comment_content']))
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, truelen_cut('-> ' + str(i) + '. ' + datalist[i]['comment_content'].splitlines()[0], self.content_width + len('-> ' + str(i))), curses.color_pair(2))
                    self.addstr(step + 10, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i]['comment_content'].split(':', 1)[0] + ':', curses.color_pair(2))
                    self.addstr(step + 12, self.startcol + (len(str(i)) + 2), break_str(datalist[i]['comment_content'].split(':', 1)[1][1:], self.startcol + (len(str(i)) + 2), maxlength), curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, truelen_cut(str(i) + '. ' + datalist[i]['comment_content'].splitlines()[0], self.content_width))
        elif datatype == 'artists':
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i]['artists_name'] + self.space + str(datalist[i]['alias']), curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, str(i) + '. ' + datalist[i]['artists_name'] + self.space + datalist[i]['alias'])
        elif datatype == 'artist_info':
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i]['item'], curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, str(i) + '. ' + datalist[i]['item'])
        elif datatype == 'albums':
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i]['albums_name'] + self.space + datalist[i]['artists_name'], curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, str(i) + '. ' + datalist[i]['albums_name'] + self.space + datalist[i]['artists_name'])
        elif datatype == 'recommend_lists':
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i]['title'], curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, str(i) + '. ' + datalist[i]['title'])
        elif datatype in ('top_playlists', 'playlists'):
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i]['playlist_name'] + self.space + datalist[i]['creator_name'], curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, str(i) + '. ' + datalist[i]['playlist_name'] + self.space + datalist[i]['creator_name'])
        elif datatype in ('toplists', 'playlist_classes', 'playlist_class_detail'):
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i], curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, str(i) + '. ' + datalist[i])
        elif datatype == 'djRadios':
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 8, self.indented_startcol, '-> ' + str(i) + '. ' + datalist[i]['name'], curses.color_pair(2))
                else:
                    self.addstr(i - offset + 8, self.startcol, str(i) + '. ' + datalist[i]['name'])
        elif datatype == 'search':
            self.screen.move(6, 1)
            self.screen.clrtobot()
            self.screen.timeout(-1)
            self.addstr(8, self.startcol, '选择搜索类型:', curses.color_pair(1))
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 10, self.indented_startcol, '-> ' + str(i) + '.' + datalist[i - 1], curses.color_pair(2))
                else:
                    self.addstr(i - offset + 10, self.startcol, str(i) + '.' + datalist[i - 1])
            self.screen.timeout(100)
        elif datatype == 'help':
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(i - offset + 9, self.indented_startcol, "-> {}. '{}{}   {}".format(i, (datalist[i][0] + "'").ljust(11), datalist[i][1].ljust(16), datalist[i][2]), curses.color_pair(2))
                else:
                    self.addstr(i - offset + 9, self.startcol, "{}. '{}{}   {}".format(i, (datalist[i][0] + "'").ljust(11), datalist[i][1].ljust(16), datalist[i][2]))
            self.addstr(20, self.startcol, 'NetEase-MusicBox 基于Python，所有版权音乐来源于网易，本地不做任何保存')
            self.addstr(21, self.startcol, '按 [G] 到 Github 了解更多信息，帮助改进，或者Star表示支持~~')
            self.addstr(22, self.startcol, 'Build with love to music by omi')
        self.screen.refresh()

    def build_login(self):
        curses.curs_set(0)
        self.build_login_bar()
        account = self.get_account()
        password = md5(self.get_password())
        return (account, password)

    def build_login_bar(self):
        curses.curs_set(0)
        curses.noecho()
        self.screen.move(4, 1)
        self.screen.clrtobot()
        self.addstr(5, self.startcol, '请输入登录信息(支持手机登录)', curses.color_pair(1))
        self.addstr(8, self.startcol, '账号:', curses.color_pair(1))
        self.addstr(9, self.startcol, '密码:', curses.color_pair(1))
        self.screen.move(8, 24)
        self.screen.refresh()

    def build_login_error(self):
        curses.curs_set(0)
        self.screen.move(4, 1)
        self.screen.timeout(-1)
        self.screen.clrtobot()
        self.addstr(8, self.startcol, '艾玛，登录信息好像不对呢 (O_O)#', curses.color_pair(1))
        self.addstr(10, self.startcol, '[1] 再试一次')
        self.addstr(11, self.startcol, '[2] 稍后再试')
        self.addstr(14, self.startcol, '请键入对应数字:', curses.color_pair(2))
        self.screen.refresh()
        x = self.screen.getch()
        self.screen.timeout(100)
        return x

    def build_search_error(self):
        curses.curs_set(0)
        self.screen.move(4, 1)
        self.screen.timeout(-1)
        self.screen.clrtobot()
        self.addstr(8, self.startcol, '是不支持的搜索类型呢...', curses.color_pair(3))
        self.addstr(9, self.startcol, '（在做了，在做了，按任意键关掉这个提示）', curses.color_pair(3))
        self.screen.refresh()
        x = self.screen.getch()
        self.screen.timeout(100)
        return x

    def build_timing(self):
        curses.curs_set(0)
        self.screen.move(6, 1)
        self.screen.clrtobot()
        self.screen.timeout(-1)
        self.addstr(8, self.startcol, '输入定时时间(min):', curses.color_pair(1))
        self.addstr(11, self.startcol, 'ps:定时时间为整数，输入0代表取消定时退出', curses.color_pair(1))
        self.screen.timeout(-1)
        curses.echo()
        timing_time = self.screen.getstr(8, self.startcol + 19, 60)
        self.screen.timeout(100)
        return timing_time

    def get_account(self):
        self.screen.timeout(-1)
        curses.echo()
        account = self.screen.getstr(8, self.startcol + 6, 60)
        self.screen.timeout(100)
        return account.decode('utf-8')

    def get_password(self):
        self.screen.timeout(-1)
        curses.noecho()
        password = self.screen.getstr(9, self.startcol + 6, 60)
        self.screen.timeout(100)
        return password.decode('utf-8')

    def get_param(self, prompt_string):
        curses.echo()
        self.screen.move(4, 1)
        self.screen.clrtobot()
        self.addstr(5, self.startcol, prompt_string, curses.color_pair(1))
        self.screen.refresh()
        keyword = self.screen.getstr(10, self.startcol, 60)
        return keyword.decode('utf-8').strip()

    def update_size(self):
        curses.curs_set(0)
        size = get_terminal_size()
        x = size[0]
        y = size[1]
        if (x, y) == (self.x, self.y):
            return
        (self.x, self.y) = (x, y)
        curses.resizeterm(self.y, self.x)
        self.update_margin()
        self.update_space()
        self.screen.clear()
        self.screen.refresh()

    def update_space(self):
        if self.x > 140:
            self.space = '   -   '
        elif self.x > 80:
            self.space = '  -  '
        else:
            self.space = ' - '
        self.screen.refresh()

# File: NEMbox\utils.py

"""
定义几个函数 写文件 通知 返回键 创建目录 创建文件
"""
import os
import platform
import subprocess
import hashlib
from collections import OrderedDict
__all__ = ['utf8_data_to_file', 'notify', 'uniq', 'create_dir', 'create_file', 'md5']

def md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def mkdir(path):
    try:
        os.mkdir(path)
        return True
    except OSError:
        return False

def create_dir(path):
    if not os.path.exists(path):
        return mkdir(path)
    elif os.path.isdir(path):
        return True
    else:
        os.remove(path)
        return mkdir(path)

def create_file(path, default='\n'):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(default)

def uniq(arr):
    return list(OrderedDict.fromkeys(arr).keys())

def utf8_data_to_file(f, data):
    if hasattr(data, 'decode'):
        f.write(data.decode('utf-8'))
    else:
        f.write(data)

def notify_command_osx(msg, msg_type, duration_time=None):
    command = ['/usr/bin/osascript', '-e']
    tpl = 'display notification "{}" {} with title "musicbox"'
    sound = 'sound name "/System/Library/Sounds/Ping.aiff"' if msg_type else ''
    command.append(tpl.format(msg, sound).encode('utf-8'))
    return command

def notify_command_linux(msg, duration_time=None):
    command = ['/usr/bin/notify-send']
    command.append(msg.encode('utf-8'))
    if duration_time:
        command.extend(['-t', str(duration_time)])
    command.extend(['-h', 'int:transient:1'])
    return command

def notify(msg, msg_type=0, duration_time=None):
    """Show system notification with duration t (ms)"""
    msg = msg.replace('"', '\\"')
    if platform.system() == 'Darwin':
        command = notify_command_osx(msg, msg_type, duration_time)
    else:
        command = notify_command_linux(msg, duration_time)
    try:
        subprocess.call(command)
        return True
    except OSError:
        return False
if __name__ == '__main__':
    notify('I\'m test ""quote', msg_type=1, duration_time=1000)
    notify("I'm test 1", msg_type=1, duration_time=1000)
    notify("I'm test 2", msg_type=0, duration_time=1000)

# File: tests\__init__.py



# File: tests\test_api.py

from NEMbox.api import NetEase
from NEMbox.api import Parse

def test_api():
    api = NetEase()
    ids = [347230, 496619464, 405998841, 28012031]
    print(api.songs_url(ids))
    print(api.songs_detail(ids))
    print(Parse.song_url(api.songs_detail(ids)[0]))
    print(api.songs_url([561307346]))