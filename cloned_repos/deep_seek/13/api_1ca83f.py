"""
网易云音乐 Api
"""
import json
import platform
import time
from collections import OrderedDict
from http.cookiejar import Cookie
from http.cookiejar import MozillaCookieJar
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
import requests_cache
from .config import Config
from .const import Constant
from .encrypt import encrypted_request
from .logger import getLogger
from .storage import Storage

requests_cache.install_cache(Constant.cache_path, expire_after=3600)
log = getLogger(__name__)

TOP_LIST_ALL: Dict[int, List[str]] = {
    0: ['云音乐新歌榜', '3779629'],
    1: ['云音乐热歌榜', '3778678'],
    2: ['网易原创歌曲榜', '2884035'],
    3: ['云音乐飙升榜', '19723756'],
    4: ['云音乐电音榜', '10520166'],
    5: ['UK排行榜周榜', '180106'],
    6: ['美国Billboard周榜', '60198'],
    7: ['KTV嗨榜', '21845217'],
    8: ['iTunes榜', '11641012'],
    9: ['Hit FM Top榜', '120001'],
    10: ['日本Oricon周榜', '60131'],
    11: ['韩国Melon排行榜周榜', '3733003'],
    12: ['韩国Mnet排行榜周榜', '60255'],
    13: ['韩国Melon原声周榜', '46772709'],
    14: ['中国TOP排行榜(港台榜)', '112504'],
    15: ['中国TOP排行榜(内地榜)', '64016'],
    16: ['香港电台中文歌曲龙虎榜', '10169002'],
    17: ['华语金曲榜', '4395559'],
    18: ['中国嘻哈榜', '1899724'],
    19: ['法国 NRJ EuroHot 30周榜', '27135204'],
    20: ['台湾Hito排行榜', '112463'],
    21: ['Beatport全球电子舞曲榜', '3812895'],
    22: ['云音乐ACG音乐榜', '71385702'],
    23: ['云音乐嘻哈榜', '991319590']
}

PLAYLIST_CLASSES: OrderedDict[str, List[str]] = OrderedDict([
    ('语种', ['华语', '欧美', '日语', '韩语', '粤语', '小语种']),
    ('风格', ['流行', '摇滚', '民谣', '电子', '舞曲', '说唱', '轻音乐', '爵士', '乡村', 'R&B/Soul', '古典', '民族', '英伦', '金属', '朋克', '蓝调', '雷鬼', '世界音乐', '拉丁', '另类/独立', 'New Age', '古风', '后摇', 'Bossa Nova']),
    ('场景', ['清晨', '夜晚', '学习', '工作', '午休', '下午茶', '地铁', '驾车', '运动', '旅行', '散步', '酒吧']),
    ('情感', ['怀旧', '清新', '浪漫', '性感', '伤感', '治愈', '放松', '孤独', '感动', '兴奋', '快乐', '安静', '思念']),
    ('主题', ['影视原声', 'ACG', '儿童', '校园', '游戏', '70后', '80后', '90后', '网络歌曲', 'KTV', '经典', '翻唱', '吉他', '钢琴', '器乐', '榜单', '00后'])
])

DEFAULT_TIMEOUT: int = 10
BASE_URL: str = 'http://music.163.com'

class Parse:

    @classmethod
    def _song_url_by_id(cls, sid: int) -> Tuple[str, str]:
        url: str = f'http://music.163.com/song/media/outer/url?id={sid}.mp3'
        quality: str = 'LD 128k'
        return (url, quality)

    @classmethod
    def song_url(cls, song: Dict[str, Any]) -> Tuple[str, str]:
        if 'url' in song:
            url: Optional[str] = song['url']
            if url is None:
                return cls._song_url_by_id(song['id'])
            br: int = song['br']
            if br >= 320000:
                quality: str = 'HD'
            elif br >= 192000:
                quality = 'MD'
            else:
                quality = 'LD'
            return (url, f'{quality} {br // 1000}k')
        else:
            return cls._song_url_by_id(song['id'])

    @classmethod
    def song_album(cls, song: Dict[str, Any]) -> Tuple[str, Union[int, str]]:
        album_name: str
        album_id: Union[int, str]
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
            raise ValueError("Invalid song data")
        return (album_name, album_id)

    @classmethod
    def song_artist(cls, song: Dict[str, Any]) -> str:
        artist: str = ''
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
    def songs(cls, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        song_info_list: List[Dict[str, Any]] = []
        for song in songs:
            url, quality = cls.song_url(song)
            if not url:
                continue
            album_name, album_id = cls.song_album(song)
            song_info: Dict[str, Any] = {
                'song_id': song['id'],
                'artist': cls.song_artist(song),
                'song_name': song['name'],
                'album_name': album_name,
                'album_id': album_id,
                'mp3_url': url,
                'quality': quality,
                'expires': song.get('expires', 0),
                'get_time': song.get('get_time', 0)
            }
            song_info_list.append(song_info)
        return song_info_list

    @classmethod
    def artists(cls, artists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            'artist_id': artist['id'],
            'artists_name': artist['name'],
            'alias': ''.join(artist.get('alias', []))
        } for artist in artists]

    @classmethod
    def albums(cls, albums: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            'album_id': album['id'],
            'albums_name': album['name'],
            'artists_name': album['artist']['name']
        } for album in albums]

    @classmethod
    def playlists(cls, playlists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            'playlist_id': pl['id'],
            'playlist_name': pl['name'],
            'creator_name': pl['creator']['nickname']
        } for pl in playlists]

class NetEase:

    def __init__(self) -> None:
        self.header: Dict[str, str] = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip,deflate,sdch',
            'Accept-Language': 'zh-CN,zh;q=0.8,gl;q=0.6,zh-TW;q=0.4',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Host': 'music.163.com',
            'Referer': 'http://music.163.com',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.87 Safari/537.36'
        }
        self.storage: Storage = Storage()
        cookie_jar: MozillaCookieJar = MozillaCookieJar(self.storage.cookie_path)
        cookie_jar.load()
        self.session: requests.Session = requests.Session()
        self.session.cookies = cookie_jar
        for cookie in cookie_jar:
            if cookie.is_expired():
                cookie_jar.clear()
                self.storage.database['user'] = {
                    'username': '',
                    'password': '',
                    'user_id': '',
                    'nickname': ''
                }
                self.storage.save()
                break

    @property
    def toplists(self) -> List[str]:
        return [item[0] for item in TOP_LIST_ALL.values()]

    def logout(self) -> None:
        self.session.cookies.clear()
        self.storage.database['user'] = {
            'username': '',
            'password': '',
            'user_id': '',
            'nickname': ''
        }
        self.session.cookies.save()
        self.storage.save()

    def _raw_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        resp: Optional[requests.Response] = None
        if method == 'GET':
            resp = self.session.get(endpoint, params=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        elif method == 'POST':
            resp = self.session.post(endpoint, data=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        return resp

    def make_cookie(self, name: str, value: str) -> Cookie:
        return Cookie(
            version=0,
            name=name,
            value=value,
            port=None,
            port_specified=False,
            domain='music.163.com',
            domain_specified=True,
            domain_initial_dot=False,
            path='/',
            path_specified=True,
            secure=False,
            expires=None,
            discard=False,
            comment=None,
            comment_url=None,
            rest={}
        )

    def request(self, method: str, path: str, params: Dict[str, Any] = {}, default: Dict[str, Any] = {'code': -1}, custom_cookies: Dict[str, str] = {}) -> Dict[str, Any]:
        endpoint: str = f'{BASE_URL}{path}'
        csrf_token: str = ''
        for cookie in self.session.cookies:
            if cookie.name == '__csrf':
                csrf_token = cookie.value
                break
        params.update({'csrf_token': csrf_token})
        data: Dict[str, Any] = default
        for key, value in custom_cookies.items():
            cookie: Cookie = self.make_cookie(key, value)
            self.session.cookies.set_cookie(cookie)
        params = encrypted_request(params)
        resp: Optional[requests.Response] = None
        try:
            resp = self._raw_request(method, endpoint, params)
            data = resp.json()
        except requests.exceptions.RequestException as e:
            log.error(e)
        except ValueError:
            log.error(f'Path: {path}, response: {resp.text[:200] if resp else "No response"}')
        finally:
            return data

    def login(self, username: str, password: str) -> Dict[str, Any]:
        self.session.cookies.load()
        if username.isdigit():
            path: str = '/weapi/login/cellphone'
            params: Dict[str, Any] = {
                'phone': username,
                'password': password,
                'countrycode': '86',
                'rememberLogin': 'true'
            }
        else:
            path = '/weapi/login'
            params = {
                'username': username,
                'password': password,
                'rememberLogin': 'true'
            }
        data: Dict[str, Any] = self.request('POST', path, params, custom_cookies={'os': 'pc'})
        self.session.cookies.save()
        return data

    def daily_task(self, is_mobile: bool = True) -> Dict[str, Any]:
        path: str = '/weapi/point/dailyTask'
        params: Dict[str, Any] = {'type': 0 if is_mobile else 1}
        return self.request('POST', path, params)

    def user_playlist(self, uid: Union[int, str], offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = '/weapi/user/playlist'
        params: Dict[str, Any] = {'uid': uid, 'offset': offset, 'limit': limit}
        return self.request('POST', path, params).get('playlist', [])

    def recommend_resource(self) -> List[Dict[str, Any]]:
        path: str = '/weapi/v1/discovery/recommend/resource'
        return self.request('POST', path).get('recommend', [])

    def recommend_playlist(self, total: bool = True, offset: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        path: str = '/weapi/v1/discovery/recommend/songs'
        params: Dict[str, Any] = {'total': total, 'offset': offset, 'limit': limit}
        return self.request('POST', path, params).get('recommend', [])

    def personal_fm(self) -> List[Dict[str, Any]]:
        path: str = '/weapi/v1/radio/get'
        return self.request('POST', path).get('data', [])

    def fm_like(self, songid: int, like: bool = True, time: int = 25, alg: str = 'itembased') -> bool:
        path: str = '/weapi/radio/like'
        params: Dict[str, Any] = {
            'alg': alg,
            'trackId': songid,
            'like': 'true' if like else 'false',
            'time': time
        }
        return self.request('POST', path, params)['code'] == 200

    def fm_trash(self, songid: int, time: int = 25, alg: str = 'RT') -> bool:
        path: str = '/weapi/radio/trash/add'
        params: Dict[str, Any] = {
            'songId': songid,
            'alg': alg,
            'time': time
        }
        return self.request('POST', path, params)['code'] == 200

    def search(self, keywords: str, stype: int = 1, offset: int = 0, total: str = 'true', limit: int = 50) -> Dict[str, Any]:
        path: str = '/weapi/search/get'
        params: Dict[str, Any] = {
            's': keywords,
            'type': stype,
            'offset': offset,
            'total': total,
            'limit': limit
        }
        return self.request('POST', path, params).get('result', {})

    def new_albums(self, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = '/weapi/album/new'
        params: Dict[str, Any] = {
            'area': 'ALL',
            'offset': offset,
            'total': True,
            'limit': limit
        }
        return self.request('POST', path, params).get('albums', [])

    def top_playlists(self, category: str = '全部', order: str = 'hot', offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = '/weapi/playlist/list'
        params: Dict[str, Any] = {
            'cat': category,
            'order': order,
            'offset': offset,
            'total': 'true',
            'limit': limit
        }
        return self.request('POST', path, params).get('playlists', [])

    def playlist_catelogs(self) -> Dict[str, Any]:
        path: str = '/weapi/playlist/catalogue'
        return self.request('POST', path)

    def playlist_songlist(self, playlist_id: Union[int, str]) -> List[Dict[str, Any]]:
        path: str = '/weapi/v3/playlist/detail'
        params: Dict[str, Any] = {
            'id': playlist_id,
            'total': 'true',
            'limit': 1000,
            'n': 1000,
            'offest': 0
        }
        custom_cookies: Dict[str, str] = {'os': platform.system()}
        return self.request('POST', path, params, {'code': -1}, custom_cookies).get('playlist', {}).get('trackIds', [])

    def top_artists(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        path: str = '/weapi/artist/top'
        params: Dict[str, Any] = {
            'offset': offset,
            'total': True,
            'limit': limit
        }
        return self.request('POST', path, params).get('artists', [])

    def top_songlist(self, idx: int = 0, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        playlist_id: str = TOP_LIST_ALL[idx][1]
        return self.playlist_songlist(playlist_id)

    def artists(self, artist_id: Union[int, str]) -> List[Dict[str, Any]]:
        path: str = f'/weapi/v1/artist/{artist_id}'
        return self.request('POST', path).get('hotSongs', [])

    def get_artist_