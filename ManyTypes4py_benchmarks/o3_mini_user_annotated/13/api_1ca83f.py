#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: omi
# @Date:   2014-08-24 21:51:57
"""
网易云音乐 Api
"""
import json
import platform
import time
from collections import OrderedDict
from http.cookiejar import Cookie
from http.cookiejar import MozillaCookieJar
from typing import Any, Dict, List, Tuple, Union, Optional

import requests
import requests_cache

from .config import Config
from .const import Constant
from .encrypt import encrypted_request
from .logger import getLogger
from .storage import Storage

requests_cache.install_cache(Constant.cache_path, expire_after=3600)

log = getLogger(__name__)

# 歌曲榜单地址
TOP_LIST_ALL: Dict[int, List[str]] = {
    0: ["云音乐新歌榜", "3779629"],
    1: ["云音乐热歌榜", "3778678"],
    2: ["网易原创歌曲榜", "2884035"],
    3: ["云音乐飙升榜", "19723756"],
    4: ["云音乐电音榜", "10520166"],
    5: ["UK排行榜周榜", "180106"],
    6: ["美国Billboard周榜", "60198"],
    7: ["KTV嗨榜", "21845217"],
    8: ["iTunes榜", "11641012"],
    9: ["Hit FM Top榜", "120001"],
    10: ["日本Oricon周榜", "60131"],
    11: ["韩国Melon排行榜周榜", "3733003"],
    12: ["韩国Mnet排行榜周榜", "60255"],
    13: ["韩国Melon原声周榜", "46772709"],
    14: ["中国TOP排行榜(港台榜)", "112504"],
    15: ["中国TOP排行榜(内地榜)", "64016"],
    16: ["香港电台中文歌曲龙虎榜", "10169002"],
    17: ["华语金曲榜", "4395559"],
    18: ["中国嘻哈榜", "1899724"],
    19: ["法国 NRJ EuroHot 30周榜", "27135204"],
    20: ["台湾Hito排行榜", "112463"],
    21: ["Beatport全球电子舞曲榜", "3812895"],
    22: ["云音乐ACG音乐榜", "71385702"],
    23: ["云音乐嘻哈榜", "991319590"],
}

PLAYLIST_CLASSES: OrderedDict[str, List[str]] = OrderedDict(
    [
        ("语种", ["华语", "欧美", "日语", "韩语", "粤语", "小语种"]),
        (
            "风格",
            [
                "流行",
                "摇滚",
                "民谣",
                "电子",
                "舞曲",
                "说唱",
                "轻音乐",
                "爵士",
                "乡村",
                "R&B/Soul",
                "古典",
                "民族",
                "英伦",
                "金属",
                "朋克",
                "蓝调",
                "雷鬼",
                "世界音乐",
                "拉丁",
                "另类/独立",
                "New Age",
                "古风",
                "后摇",
                "Bossa Nova",
            ],
        ),
        (
            "场景",
            ["清晨", "夜晚", "学习", "工作", "午休", "下午茶", "地铁", "驾车", "运动", "旅行", "散步", "酒吧"],
        ),
        (
            "情感",
            [
                "怀旧",
                "清新",
                "浪漫",
                "性感",
                "伤感",
                "治愈",
                "放松",
                "孤独",
                "感动",
                "兴奋",
                "快乐",
                "安静",
                "思念",
            ],
        ),
        (
            "主题",
            [
                "影视原声",
                "ACG",
                "儿童",
                "校园",
                "游戏",
                "70后",
                "80后",
                "90后",
                "网络歌曲",
                "KTV",
                "经典",
                "翻唱",
                "吉他",
                "钢琴",
                "器乐",
                "榜单",
                "00后",
            ],
        ),
    ]
)

DEFAULT_TIMEOUT: int = 10

BASE_URL: str = "http://music.163.com"


class Parse(object):
    @classmethod
    def _song_url_by_id(cls, sid: int) -> Tuple[str, str]:
        # 128k
        url: str = "http://music.163.com/song/media/outer/url?id={}.mp3".format(sid)
        quality: str = "LD 128k"
        return url, quality

    @classmethod
    def song_url(cls, song: Dict[str, Any]) -> Tuple[str, str]:
        if "url" in song:
            # songs_url resp
            url: Optional[str] = song["url"]
            if url is None:
                return Parse._song_url_by_id(song["id"])
            br: int = song["br"]
            if br >= 320000:
                quality: str = "HD"
            elif br >= 192000:
                quality = "MD"
            else:
                quality = "LD"
            return url, "{} {}k".format(quality, br // 1000)
        else:
            # songs_detail resp
            return Parse._song_url_by_id(song["id"])

    @classmethod
    def song_album(cls, song: Dict[str, Any]) -> Tuple[str, Union[str, int]]:
        # 对新老接口进行处理
        if "al" in song:
            if song["al"] is not None:
                album_name: str = song["al"]["name"]
                album_id: Union[str, int] = song["al"]["id"]
            else:
                album_name = "未知专辑"
                album_id = ""
        elif "album" in song:
            if song["album"] is not None:
                album_name = song["album"]["name"]
                album_id = song["album"]["id"]
            else:
                album_name = "未知专辑"
                album_id = ""
        else:
            raise ValueError
        return album_name, album_id

    @classmethod
    def song_artist(cls, song: Dict[str, Any]) -> str:
        artist: str = ""
        # 对新老接口进行处理
        if "ar" in song:
            artist = ", ".join([a["name"] for a in song["ar"] if a["name"] is not None])
            if artist == "" and "pc" in song:
                artist = "未知艺术家" if song["pc"]["ar"] is None else song["pc"]["ar"]
        elif "artists" in song:
            artist = ", ".join([a["name"] for a in song["artists"]])
        else:
            artist = "未知艺术家"
        return artist

    @classmethod
    def songs(cls, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        song_info_list: List[Dict[str, Any]] = []
        for song in songs:
            url, quality = Parse.song_url(song)
            if not url:
                continue
            album_name, album_id = Parse.song_album(song)
            song_info: Dict[str, Any] = {
                "song_id": song["id"],
                "artist": Parse.song_artist(song),
                "song_name": song["name"],
                "album_name": album_name,
                "album_id": album_id,
                "mp3_url": url,
                "quality": quality,
                "expires": song["expires"],
                "get_time": song["get_time"],
            }
            song_info_list.append(song_info)
        return song_info_list

    @classmethod
    def artists(cls, artists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "artist_id": artist["id"],
                "artists_name": artist["name"],
                "alias": "".join(artist["alias"]),
            }
            for artist in artists
        ]

    @classmethod
    def albums(cls, albums: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "album_id": album["id"],
                "albums_name": album["name"],
                "artists_name": album["artist"]["name"],
            }
            for album in albums
        ]

    @classmethod
    def playlists(cls, playlists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "playlist_id": pl["id"],
                "playlist_name": pl["name"],
                "creator_name": pl["creator"]["nickname"],
            }
            for pl in playlists
        ]


class NetEase(object):
    def __init__(self) -> None:
        self.header: Dict[str, str] = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip,deflate,sdch",
            "Accept-Language": "zh-CN,zh;q=0.8,gl;q=0.6,zh-TW;q=0.4",
            "Connection": "keep-alive",
            "Content-Type": "application/x-www-form-urlencoded",
            "Host": "music.163.com",
            "Referer": "http://music.163.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.87 Safari/537.36",
        }
        self.storage: Storage = Storage()
        cookie_jar: MozillaCookieJar = MozillaCookieJar(self.storage.cookie_path)
        cookie_jar.load()
        self.session: requests.Session = requests.Session()
        self.session.cookies = cookie_jar
        for cookie in cookie_jar:
            if cookie.is_expired():
                cookie_jar.clear()
                self.storage.database["user"] = {
                    "username": "",
                    "password": "",
                    "user_id": "",
                    "nickname": "",
                }
                self.storage.save()
                break

    @property
    def toplists(self) -> List[str]:
        return [item[0] for item in TOP_LIST_ALL.values()]

    def logout(self) -> None:
        self.session.cookies.clear()
        self.storage.database["user"] = {
            "username": "",
            "password": "",
            "user_id": "",
            "nickname": "",
        }
        self.session.cookies.save()
        self.storage.save()

    def _raw_request(self, method: str, endpoint: str, data: Optional[Any] = None) -> requests.Response:
        resp: Optional[requests.Response] = None
        if method == "GET":
            resp = self.session.get(endpoint, params=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        elif method == "POST":
            resp = self.session.post(endpoint, data=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        # Assuming resp is not None
        return resp  # type: ignore

    # 生成Cookie对象
    def make_cookie(self, name: str, value: str) -> Cookie:
        return Cookie(
            version=0,
            name=name,
            value=value,
            port=None,
            port_specified=False,
            domain="music.163.com",
            domain_specified=True,
            domain_initial_dot=False,
            path="/",
            path_specified=True,
            secure=False,
            expires=None,
            discard=False,
            comment=None,
            comment_url=None,
            rest={},
        )

    def request(
        self,
        method: str,
        path: str,
        params: Dict[str, Any] = {},
        default: Dict[str, Any] = {"code": -1},
        custom_cookies: Dict[str, str] = {},
    ) -> Dict[str, Any]:
        endpoint: str = "{}{}".format(BASE_URL, path)
        csrf_token: str = ""
        for cookie in self.session.cookies:
            if cookie.name == "__csrf":
                csrf_token = cookie.value
                break
        params.update({"csrf_token": csrf_token})
        data: Dict[str, Any] = default

        for key, value in custom_cookies.items():
            cookie = self.make_cookie(key, value)
            self.session.cookies.set_cookie(cookie)

        encrypted_params: Dict[str, Any] = encrypted_request(params)
        resp: Optional[requests.Response] = None
        try:
            resp = self._raw_request(method, endpoint, encrypted_params)
            data = resp.json()
        except requests.exceptions.RequestException as e:
            log.error(e)
        except ValueError:
            log.error("Path: {}, response: {}".format(path, resp.text[:200] if resp else ""))
        finally:
            return data

    def login(self, username: str, password: str) -> Dict[str, Any]:
        self.session.cookies.load()
        if username.isdigit():
            path: str = "/weapi/login/cellphone"
            params: Dict[str, Any] = dict(
                phone=username,
                password=password,
                countrycode="86",
                rememberLogin="true",
            )
        else:
            path = "/weapi/login"
            params = dict(
                username=username,
                password=password,
                rememberLogin="true",
            )
        data: Dict[str, Any] = self.request("POST", path, params, custom_cookies={"os": "pc"})
        self.session.cookies.save()
        return data

    # 每日签到
    def daily_task(self, is_mobile: bool = True) -> Dict[str, Any]:
        path: str = "/weapi/point/dailyTask"
        params: Dict[str, Any] = dict(type=0 if is_mobile else 1)
        return self.request("POST", path, params)

    # 用户歌单
    def user_playlist(self, uid: Union[int, str], offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = "/weapi/user/playlist"
        params: Dict[str, Any] = dict(uid=uid, offset=offset, limit=limit)
        return self.request("POST", path, params).get("playlist", [])

    # 每日推荐歌单
    def recommend_resource(self) -> List[Dict[str, Any]]:
        path: str = "/weapi/v1/discovery/recommend/resource"
        return self.request("POST", path).get("recommend", [])

    # 每日推荐歌曲
    def recommend_playlist(self, total: bool = True, offset: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        path: str = "/weapi/v1/discovery/recommend/songs"
        params: Dict[str, Any] = dict(total=total, offset=offset, limit=limit)
        return self.request("POST", path, params).get("recommend", [])

    # 私人FM
    def personal_fm(self) -> List[Dict[str, Any]]:
        path: str = "/weapi/v1/radio/get"
        return self.request("POST", path).get("data", [])

    # like
    def fm_like(self, songid: Union[int, str], like: bool = True, time: int = 25, alg: str = "itembased") -> bool:
        path: str = "/weapi/radio/like"
        params: Dict[str, Any] = dict(
            alg=alg, trackId=songid, like="true" if like else "false", time=time
        )
        return self.request("POST", path, params)["code"] == 200

    # FM trash
    def fm_trash(self, songid: Union[int, str], time: int = 25, alg: str = "RT") -> bool:
        path: str = "/weapi/radio/trash/add"
        params: Dict[str, Any] = dict(songId=songid, alg=alg, time=time)
        return self.request("POST", path, params)["code"] == 200

    # 搜索单曲(1)，歌手(100)，专辑(10)，歌单(1000)，用户(1002)
    def search(
        self, keywords: str, stype: int = 1, offset: int = 0, total: Union[str, bool] = "true", limit: int = 50
    ) -> Dict[str, Any]:
        path: str = "/weapi/search/get"
        params: Dict[str, Any] = dict(s=keywords, type=stype, offset=offset, total=total, limit=limit)
        return self.request("POST", path, params).get("result", {})

    # 新碟上架
    def new_albums(self, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = "/weapi/album/new"
        params: Dict[str, Any] = dict(area="ALL", offset=offset, total=True, limit=limit)
        return self.request("POST", path, params).get("albums", [])

    # 歌单（网友精选碟） hot||new
    def top_playlists(self, category: str = "全部", order: str = "hot", offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = "/weapi/playlist/list"
        params: Dict[str, Any] = dict(
            cat=category, order=order, offset=offset, total="true", limit=limit
        )
        return self.request("POST", path, params).get("playlists", [])

    def playlist_catelogs(self) -> Dict[str, Any]:
        path: str = "/weapi/playlist/catalogue"
        return self.request("POST", path)

    # 歌单详情
    def playlist_songlist(self, playlist_id: Union[int, str]) -> List[Dict[str, Any]]:
        path: str = "/weapi/v3/playlist/detail"
        params: Dict[str, Any] = dict(id=playlist_id, total="true", limit=1000, n=1000, offest=0)
        custom_cookies: Dict[str, str] = dict(os=platform.system())
        return (
            self.request("POST", path, params, {"code": -1}, custom_cookies)
            .get("playlist", {})
            .get("trackIds", [])
        )

    # 热门歌手
    def top_artists(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        path: str = "/weapi/artist/top"
        params: Dict[str, Any] = dict(offset=offset, total=True, limit=limit)
        return self.request("POST", path, params).get("artists", [])

    # 热门单曲
    def top_songlist(self, idx: int = 0, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        playlist_id: str = TOP_LIST_ALL[idx][1]
        return self.playlist_songlist(playlist_id)

    # 歌手单曲
    def artists(self, artist_id: Union[int, str]) -> List[Dict[str, Any]]:
        path: str = "/weapi/v1/artist/{}".format(artist_id)
        return self.request("POST", path).get("hotSongs", [])

    def get_artist_album(self, artist_id: Union[int, str], offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = "/weapi/artist/albums/{}".format(artist_id)
        params: Dict[str, Any] = dict(offset=offset, total=True, limit=limit)
        return self.request("POST", path, params).get("hotAlbums", [])

    # album id --> song id set
    def album(self, album_id: Union[int, str]) -> List[Dict[str, Any]]:
        path: str = "/weapi/v1/album/{}".format(album_id)
        return self.request("POST", path).get("songs", [])

    def song_comments(self, music_id: Union[int, str], offset: int = 0, total: Union[str, bool] = "false", limit: int = 100) -> Dict[str, Any]:
        path: str = "/weapi/v1/resource/comments/R_SO_4_{}/".format(music_id)
        params: Dict[str, Any] = dict(rid=music_id, offset=offset, total=total, limit=limit)
        return self.request("POST", path, params)

    # song ids --> song urls ( details )
    def songs_detail(self, ids: List[Union[int, str]]) -> List[Dict[str, Any]]:
        path: str = "/weapi/v3/song/detail"
        params: Dict[str, Any] = dict(c=json.dumps([{"id": _id} for _id in ids]), ids=json.dumps(ids))
        return self.request("POST", path, params).get("songs", [])

    def songs_url(self, ids: List[Union[int, str]]) -> List[Dict[str, Any]]:
        quality: int = Config().get("music_quality")
        rate_map: Dict[int, int] = {0: 320000, 1: 192000, 2: 128000}
        path: str = "/weapi/song/enhance/player/url"
        params: Dict[str, Any] = dict(ids=ids, br=rate_map[quality])
        return self.request("POST", path, params).get("data", [])

    # lyric
    def song_lyric(self, music_id: Union[int, str]) -> List[str]:
        path: str = "/weapi/song/lyric"
        params: Dict[str, Any] = dict(os="osx", id=music_id, lv=-1, kv=-1, tv=-1)
        lyric: Any = self.request("POST", path, params).get("lrc", {}).get("lyric", [])
        if not lyric:
            return []
        else:
            return lyric.split("\n")

    def song_tlyric(self, music_id: Union[int, str]) -> List[str]:
        path: str = "/weapi/song/lyric"
        params: Dict[str, Any] = dict(os="osx", id=music_id, lv=-1, kv=-1, tv=-1)
        lyric: Any = self.request("POST", path, params).get("tlyric", {}).get("lyric", [])
        if not lyric:
            return []
        else:
            return lyric.split("\n")

    # 今日最热（0）, 本周最热（10），历史最热（20），最新节目（30）
    def djRadios(self, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = "/weapi/djradio/hot/v1"
        params: Dict[str, Any] = dict(limit=limit, offset=offset)
        return self.request("POST", path, params).get("djRadios", [])

    def djprograms(self, radio_id: Union[int, str], asc: bool = False, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path: str = "/weapi/dj/program/byradio"
        params: Dict[str, Any] = dict(asc=asc, radioId=radio_id, offset=offset, limit=limit)
        programs: List[Dict[str, Any]] = self.request("POST", path, params).get("programs", [])
        return [p["mainSong"] for p in programs]

    def alldjprograms(self, radio_id: Union[int, str], asc: bool = False, offset: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        programs: List[Dict[str, Any]] = []
        ps: List[Dict[str, Any]] = self.djprograms(radio_id, asc=asc, offset=offset, limit=limit)
        while ps:
            programs.extend(ps)
            offset += limit
            ps = self.djprograms(radio_id, asc=asc, offset=offset, limit=limit)
        return programs

    # 获取版本
    def get_version(self) -> Dict[str, Any]:
        action: str = "https://pypi.org/pypi/NetEase-MusicBox/json"
        try:
            return requests.get(action).json()
        except requests.exceptions.RequestException as e:
            log.error(e)
            return {}

    def dig_info(self, data: Any, dig_type: str) -> Any:
        if not data:
            return []
        if dig_type in ("songs", "fmsongs", "djprograms"):
            sids: List[Any] = [x["id"] for x in data]
            urls: List[Dict[str, Any]] = []
            for i in range(0, len(sids), 350):
                urls.extend(self.songs_url(sids[i : i + 350]))
            sds: List[Any] = []
            if dig_type == "djprograms":
                sds.extend(data)
            else:
                for i in range(0, len(sids), 500):
                    sds.extend(self.songs_detail(sids[i : i + 500]))
            url_id_index: Dict[Any, int] = {}
            for index, url in enumerate(urls):
                url_id_index[url["id"]] = index
            timestamp: float = time.time()
            for s in sds:
                url_index: Optional[int] = url_id_index.get(s["id"])
                if url_index is None:
                    log.error("can't get song url, id: %s", s["id"])
                    return []
                s["url"] = urls[url_index]["url"]
                s["br"] = urls[url_index]["br"]
                s["expires"] = urls[url_index]["expi"]
                s["get_time"] = timestamp
            return Parse.songs(sds)
        elif dig_type == "refresh_urls":
            urls_info: List[Dict[str, Any]] = []
            for i in range(0, len(data), 350):
                urls_info.extend(self.songs_url(data[i : i + 350]))
            timestamp = time.time()
            songs: List[Dict[str, Any]] = []
            for url_info in urls_info:
                song: Dict[str, Any] = {}
                song["song_id"] = url_info["id"]
                song["mp3_url"] = url_info["url"]
                song["expires"] = url_info["expi"]
                song["get_time"] = timestamp
                songs.append(song)
            return songs
        elif dig_type == "artists":
            return Parse.artists(data)
        elif dig_type == "albums":
            return Parse.albums(data)
        elif dig_type in ("playlists", "top_playlists"):
            return Parse.playlists(data)
        elif dig_type == "playlist_classes":
            return list(PLAYLIST_CLASSES.keys())
        elif dig_type == "playlist_class_detail":
            return PLAYLIST_CLASSES[data]
        elif dig_type == "djRadios":
            return data
        else:
            raise ValueError("Invalid dig type")