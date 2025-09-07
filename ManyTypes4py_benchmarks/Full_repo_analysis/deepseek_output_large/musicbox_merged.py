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

def create_config() -> None:
    create_dir(Constant.conf_dir)
    create_dir(Constant.download_dir)
    create_file(Constant.storage_path)
    create_file(Constant.log_path, default='')
    create_file(Constant.cookie_path, default='# Netscape HTTP Cookie File\n')
create_config()

import _curses
import argparse
import curses
import sys
import traceback
from . import __version__
from .menu import Menu

def start() -> None:
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
    def _song_url_by_id(cls, sid: int) -> tuple[str, str]:
        url = 'http://music.163.com/song/media/outer/url?id={}.mp3'.format(sid)
        quality = 'LD 128k'
        return (url, quality)

    @classmethod
    def song_url(cls, song: dict) -> tuple[str, str]:
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
    def song_album(cls, song: dict) -> tuple[str, str]:
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
    def song_artist(cls, song: dict) -> str:
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
    def songs(cls, songs: list) -> list[dict]:
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
    def artists(cls, artists: list) -> list[dict]:
        return [{'artist_id': artist['id'], 'artists_name': artist['name'], 'alias': ''.join(artist['alias'])} for artist in artists]

    @classmethod
    def albums(cls, albums: list) -> list[dict]:
        return [{'album_id': album['id'], 'albums_name': album['name'], 'artists_name': album['artist']['name']} for album in albums]

    @classmethod
    def playlists(cls, playlists: list) -> list[dict]:
        return [{'playlist_id': pl['id'], 'playlist_name': pl['name'], 'creator_name': pl['creator']['nickname']} for pl in playlists]

class NetEase(object):

    def __init__(self) -> None:
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
    def toplists(self) -> list[str]:
        return [item[0] for item in TOP_LIST_ALL.values()]

    def logout(self) -> None:
        self.session.cookies.clear()
        self.storage.database['user'] = {'username': '', 'password': '', 'user_id': '', 'nickname': ''}
        self.session.cookies.save()
        self.storage.save()

    def _raw_request(self, method: str, endpoint: str, data: dict | None = None) -> requests.Response | None:
        resp = None
        if method == 'GET':
            resp = self.session.get(endpoint, params=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        elif method == 'POST':
            resp = self.session.post(endpoint, data=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        return resp

    def make_cookie(self, name: str, value: str) -> Cookie:
        return Cookie(version=0, name=name, value=value, port=None, port_specified=False, domain='music.163.com', domain_specified=True, domain_initial_dot=False, path='/', path_specified=True, secure=False, expires=None, discard=False, comment=None, comment_url=None, rest={})

    def request(self, method: str, path: str, params: dict = {}, default: dict = {'code': -1}, custom_cookies: dict = {}) -> dict:
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

    def login(self, username: str, password: str) -> dict:
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

    def daily_task(self, is_mobile: bool = True) -> dict:
        path = '/weapi/point/dailyTask'
        params = dict(type=0 if is_mobile else 1)
        return self.request('POST', path, params)

    def user_playlist(self, uid: int, offset: int = 0, limit: int = 50) -> list:
        path = '/weapi/user/playlist'
        params = dict(uid=uid, offset=offset, limit=limit)
        return self.request('POST', path, params).get('playlist', [])

    def recommend_resource(self) -> list:
        path = '/weapi/v1/discovery/recommend/resource'
        return self.request('POST', path).get('recommend', [])

    def recommend_playlist(self, total: bool = True, offset: int = 0, limit: int = 20) -> list:
        path = '/weapi/v1/discovery/recommend/songs'
        params = dict(total=total, offset=offset, limit=limit)
        return self.request('POST', path, params).get('recommend', [])

    def personal_fm(self) -> list:
        path = '/weapi/v1/radio/get'
        return self.request('POST', path).get('data', [])

    def fm_like(self, songid: int, like: bool = True, time: int = 25, alg: str = 'itembased') -> bool:
        path = '/weapi/radio/like'
        params = dict(alg=alg, trackId=songid, like='true' if like else 'false', time=time)
        return self.request('POST', path, params)['code'] == 200

    def fm_trash(self, songid: int, time: int = 25, alg: str = 'RT') -> bool:
        path = '/weapi/radio/trash/add'
        params = dict(songId=songid, alg=alg, time=time)
        return self.request('POST', path, params)['code'] == 200

    def search(self, keywords: str, stype: int = 1, offset: int = 0, total: str = 'true', limit: int = 50) -> dict:
        path = '/weapi/search/get'
        params = dict(s=keywords, type=stype, offset=offset, total=total, limit=limit)
        return self.request('POST', path, params).get('result', {})

    def new_albums(self, offset: int = 0, limit: int = 50) -> list:
        path = '/weapi/album/new'
        params = dict(area='ALL', offset=offset, total=True, limit=limit)
        return self.request('POST', path, params).get('albums', [])

    def top_playlists(self, category: str = '全部', order: str = 'hot', offset: int = 0, limit: int = 50) -> list:
        path = '/weapi/playlist/list'
        params = dict(cat=category, order=order, offset=offset, total='true', limit=limit)
        return self.request('POST', path, params).get('playlists', [])

    def playlist_catelogs(self) -> dict:
        path = '/weapi/playlist/catalogue'
        return self.request('POST', path)

    def playlist_songlist(self, playlist_id: int) -> list:
        path = '/weapi/v3/playlist/detail'
        params = dict(id=playlist_id, total='true', limit=1000, n=1000, offest=0)
        custom_cookies = dict(os=platform.system())
        return self.request('POST', path, params, {'code': -1}, custom_cookies).get('playlist', {}).get('trackIds', [])

    def top_artists(self, offset: int = 0, limit: int = 100) -> list:
        path = '/weapi/artist/top'
        params = dict(offset=offset, total=True, limit=limit)
        return self.request('POST', path, params).get('artists', [])

    def top_songlist(self, idx: int = 0, offset: int = 0, limit: int = 100) -> list:
        playlist_id = TOP_LIST_ALL[idx][1]
        return self.playlist_songlist(playlist_id)

    def artists(self, artist_id: int) -> list:
        path = '/weapi/v1/artist/{}'.format(artist_id)
       