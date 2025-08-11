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
    def _song_url_by_id(cls: Union[str, typing.Type, None], sid: str) -> tuple[str]:
        url = 'http://music.163.com/song/media/outer/url?id={}.mp3'.format(sid)
        quality = 'LD 128k'
        return (url, quality)

    @classmethod
    def song_url(cls: Union[str, list[dict]], song: str) -> tuple[typing.Union[str,None,list]]:
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
    def song_album(cls: str, song: str) -> tuple[typing.Text]:
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
    def song_artist(cls: Union[str, list[str], bool], song: str) -> typing.Text:
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
    def songs(cls: Union[str, list[str]], songs: Union[str, list[str]]) -> list[dict[typing.Text, typing.Union[typing.Text,list[str],set,int,dict,list]]]:
        song_info_list = []
        for song in songs:
            url, quality = Parse.song_url(song)
            if not url:
                continue
            album_name, album_id = Parse.song_album(song)
            song_info = {'song_id': song['id'], 'artist': Parse.song_artist(song), 'song_name': song['name'], 'album_name': album_name, 'album_id': album_id, 'mp3_url': url, 'quality': quality, 'expires': song['expires'], 'get_time': song['get_time']}
            song_info_list.append(song_info)
        return song_info_list

    @classmethod
    def artists(cls: Union[str, list[dict[str, typing.Any]], list[dict]], artists: Union[dict[str, dict[str, typing.Any]], list, list[str]]) -> list[dict[typing.Text, str]]:
        return [{'artist_id': artist['id'], 'artists_name': artist['name'], 'alias': ''.join(artist['alias'])} for artist in artists]

    @classmethod
    def albums(cls: list[dict], albums: str) -> list[dict[typing.Text, typing.Text]]:
        return [{'album_id': album['id'], 'albums_name': album['name'], 'artists_name': album['artist']['name']} for album in albums]

    @classmethod
    def playlists(cls: str, playlists: Union[list[str], dict[str, typing.Any], str]) -> list[dict[typing.Text, str]]:
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
    def toplists(self) -> list[typing.Union[typing.Any,str,None]]:
        return [item[0] for item in TOP_LIST_ALL.values()]

    def logout(self) -> None:
        self.session.cookies.clear()
        self.storage.database['user'] = {'username': '', 'password': '', 'user_id': '', 'nickname': ''}
        self.session.cookies.save()
        self.storage.save()

    def _raw_request(self, method: Union[str, list, dict], endpoint: Union[str, dict, None], data: Union[None, str, dict]=None) -> Union[None, typing.Generator[typing.Optional[typing.Any]], requests.models.Response]:
        resp = None
        if method == 'GET':
            resp = self.session.get(endpoint, params=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        elif method == 'POST':
            resp = self.session.post(endpoint, data=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        return resp

    def make_cookie(self, name: Union[str, None], value: Union[str, None]) -> Cookie:
        return Cookie(version=0, name=name, value=value, port=None, port_specified=False, domain='music.163.com', domain_specified=True, domain_initial_dot=False, path='/', path_specified=True, secure=False, expires=None, discard=False, comment=None, comment_url=None, rest={})

    def request(self, method: Union[dict, str, bool], path: Union[str, dict, list[str]], params: dict={}, default: dict[typing.Text, int]={'code': -1}, custom_cookies: dict={}) -> Union[list[tuple[typing.Union[typing.Any,str]]], dict[str, typing.Any], list]:
        endpoint = '{}{}'.format(BASE_URL, path)
        csrf_token = ''
        for cookie in self.session.cookies:
            if cookie.name == '__csrf':
                csrf_token = cookie.value
                break
        params.update({'csrf_token': csrf_token})
        data = default
        for key, value in custom_cookies.items():
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

    def login(self, username: str, password: Union[str, None]) -> Union[dict[str, typing.Any], dict[str, dict[str, typing.Any]], typing.Mapping, None]:
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

    def daily_task(self, is_mobile: bool=True) -> Union[list[str], typing.IO]:
        path = '/weapi/point/dailyTask'
        params = dict(type=0 if is_mobile else 1)
        return self.request('POST', path, params)

    def user_playlist(self, uid: Union[int, str, None], offset: int=0, limit: int=50) -> Union[str, bool, django.http.HttpRequest]:
        path = '/weapi/user/playlist'
        params = dict(uid=uid, offset=offset, limit=limit)
        return self.request('POST', path, params).get('playlist', [])

    def recommend_resource(self) -> Union[str, bool, tuple[typing.Union[dict[str, typing.Any],typing.Any]]]:
        path = '/weapi/v1/discovery/recommend/resource'
        return self.request('POST', path).get('recommend', [])

    def recommend_playlist(self, total: bool=True, offset: int=0, limit: int=20) -> Union[django.http.HttpResponse, str, bool]:
        path = '/weapi/v1/discovery/recommend/songs'
        params = dict(total=total, offset=offset, limit=limit)
        return self.request('POST', path, params).get('recommend', [])

    def personal_fm(self) -> str:
        path = '/weapi/v1/radio/get'
        return self.request('POST', path).get('data', [])

    def fm_like(self, songid: Union[float, str], like: bool=True, time: int=25, alg: typing.Text='itembased') -> bool:
        path = '/weapi/radio/like'
        params = dict(alg=alg, trackId=songid, like='true' if like else 'false', time=time)
        return self.request('POST', path, params)['code'] == 200

    def fm_trash(self, songid: Union[str, int, float], time: int=25, alg: typing.Text='RT') -> bool:
        path = '/weapi/radio/trash/add'
        params = dict(songId=songid, alg=alg, time=time)
        return self.request('POST', path, params)['code'] == 200

    def search(self, keywords: Union[int, None, str, list], stype: int=1, offset: int=0, total: typing.Text='true', limit: int=50) -> Union[str, dict]:
        path = '/weapi/search/get'
        params = dict(s=keywords, type=stype, offset=offset, total=total, limit=limit)
        return self.request('POST', path, params).get('result', {})

    def new_albums(self, offset: int=0, limit: int=50) -> Union[str, django.http.HttpResponse]:
        path = '/weapi/album/new'
        params = dict(area='ALL', offset=offset, total=True, limit=limit)
        return self.request('POST', path, params).get('albums', [])

    def top_playlists(self, category: typing.Text='全部', order: typing.Text='hot', offset: int=0, limit: int=50) -> Union[django.http.HttpResponse, str, bool]:
        path = '/weapi/playlist/list'
        params = dict(cat=category, order=order, offset=offset, total='true', limit=limit)
        return self.request('POST', path, params).get('playlists', [])

    def playlist_catelogs(self) -> str:
        path = '/weapi/playlist/catalogue'
        return self.request('POST', path)

    def playlist_songlist(self, playlist_id: Union[int, str, None]) -> Union[str, django.http.HttpResponse, bool]:
        path = '/weapi/v3/playlist/detail'
        params = dict(id=playlist_id, total='true', limit=1000, n=1000, offest=0)
        custom_cookies = dict(os=platform.system())
        return self.request('POST', path, params, {'code': -1}, custom_cookies).get('playlist', {}).get('trackIds', [])

    def top_artists(self, offset: int=0, limit: int=100) -> Union[str, django.http.HttpResponse, bool]:
        path = '/weapi/artist/top'
        params = dict(offset=offset, total=True, limit=limit)
        return self.request('POST', path, params).get('artists', [])

    def top_songlist(self, idx: int=0, offset: int=0, limit: int=100) -> Union[str, int]:
        playlist_id = TOP_LIST_ALL[idx][1]
        return self.playlist_songlist(playlist_id)

    def artists(self, artist_id) -> list[dict[typing.Text, str]]:
        path = '/weapi/v1/artist/{}'.format(artist_id)
        return self.request('POST', path).get('hotSongs', [])

    def get_artist_album(self, artist_id: Union[int, str], offset: int=0, limit: int=50) -> Union[str, bool, django.http.HttpResponse]:
        path = '/weapi/artist/albums/{}'.format(artist_id)
        params = dict(offset=offset, total=True, limit=limit)
        return self.request('POST', path, params).get('hotAlbums', [])

    def album(self, album_id: Union[int, str]) -> str:
        path = '/weapi/v1/album/{}'.format(album_id)
        return self.request('POST', path).get('songs', [])

    def song_comments(self, music_id: Union[int, typing.Mapping, None, typing.Sequence], offset: int=0, total: typing.Text='false', limit: int=100) -> Union[str, bytes, typing.Any, list[str]]:
        path = '/weapi/v1/resource/comments/R_SO_4_{}/'.format(music_id)
        params = dict(rid=music_id, offset=offset, total=total, limit=limit)
        return self.request('POST', path, params)

    def songs_detail(self, ids: str) -> Union[str, django.http.HttpResponse, bool]:
        path = '/weapi/v3/song/detail'
        params = dict(c=json.dumps([{'id': _id} for _id in ids]), ids=json.dumps(ids))
        return self.request('POST', path, params).get('songs', [])

    def songs_url(self, ids: Union[int, str, None]) -> Union[str, bool, django.http.HttpResponse]:
        quality = Config().get('music_quality')
        rate_map = {0: 320000, 1: 192000, 2: 128000}
        path = '/weapi/song/enhance/player/url'
        params = dict(ids=ids, br=rate_map[quality])
        return self.request('POST', path, params).get('data', [])

    def song_lyric(self, music_id: Union[str, int, None]) -> Union[list, list[str]]:
        path = '/weapi/song/lyric'
        params = dict(os='osx', id=music_id, lv=-1, kv=-1, tv=-1)
        lyric = self.request('POST', path, params).get('lrc', {}).get('lyric', [])
        if not lyric:
            return []
        else:
            return lyric.split('\n')

    def song_tlyric(self, music_id: Union[str, int]) -> Union[list, list[str]]:
        path = '/weapi/song/lyric'
        params = dict(os='osx', id=music_id, lv=-1, kv=-1, tv=-1)
        lyric = self.request('POST', path, params).get('tlyric', {}).get('lyric', [])
        if not lyric:
            return []
        else:
            return lyric.split('\n')

    def djRadios(self, offset: int=0, limit: int=50) -> Union[str, bool, None]:
        path = '/weapi/djradio/hot/v1'
        params = dict(limit=limit, offset=offset)
        return self.request('POST', path, params).get('djRadios', [])

    def djprograms(self, radio_id: Union[int, list[int], None], asc: bool=False, offset: int=0, limit: int=50) -> list[str]:
        path = '/weapi/dj/program/byradio'
        params = dict(asc=asc, radioId=radio_id, offset=offset, limit=limit)
        programs = self.request('POST', path, params).get('programs', [])
        return [p['mainSong'] for p in programs]

    def alldjprograms(self, radio_id: Union[int, None, str], asc: bool=False, offset: int=0, limit: int=500) -> list[str]:
        programs = []
        ps = self.djprograms(radio_id, asc=asc, offset=offset, limit=limit)
        while ps:
            programs.extend(ps)
            offset += limit
            ps = self.djprograms(radio_id, asc=asc, offset=offset, limit=limit)
        return programs

    def get_version(self) -> Union[dict, str, list[dict[str, str]]]:
        action = 'https://pypi.org/pypi/NetEase-MusicBox/json'
        try:
            return requests.get(action).json()
        except requests.exceptions.RequestException as e:
            log.error(e)
            return {}

    def dig_info(self, data: Any, dig_type: Union[str, typing.Any, None]) -> Union[list, str, int, list[dict[typing.Text, ]]]:
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
            for index, url in enumerate(urls):
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