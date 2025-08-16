from typing import List, Dict, Any

class Parse:
    @classmethod
    def _song_url_by_id(cls, sid: str) -> Tuple[str, str]:
        url = 'http://music.163.com/song/media/outer/url?id={}.mp3'.format(sid)
        quality = 'LD 128k'
        return (url, quality)

    @classmethod
    def song_url(cls, song: Dict[str, Any]) -> Tuple[str, str]:
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
    def song_album(cls, song: Dict[str, Any]) -> Tuple[str, str]:
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
    def song_artist(cls, song: Dict[str, Any]) -> str:
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
    def songs(cls, songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    def artists(cls, artists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{'artist_id': artist['id'], 'artists_name': artist['name'], 'alias': ''.join(artist['alias'])} for artist in artists]

    @classmethod
    def albums(cls, albums: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{'album_id': album['id'], 'albums_name': album['name'], 'artists_name': album['artist']['name']} for album in albums]

    @classmethod
    def playlists(cls, playlists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{'playlist_id': pl['id'], 'playlist_name': pl['name'], 'creator_name': pl['creator']['nickname']} for pl in playlists]

class NetEase:
    def __init__(self):
        self.header: Dict[str, str] = {'Accept': '*/*', 'Accept-Encoding': 'gzip,deflate,sdch', 'Accept-Language': 'zh-CN,zh;q=0.8,gl;q=0.6,zh-TW;q=0.4', 'Connection': 'keep-alive', 'Content-Type': 'application/x-www-form-urlencoded', 'Host': 'music.163.com', 'Referer': 'http://music.163.com', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.87 Safari/537.36'}
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
    def toplists(self) -> List[str]:
        return [item[0] for item in TOP_LIST_ALL.values()]

    def logout(self) -> None:
        self.session.cookies.clear()
        self.storage.database['user'] = {'username': '', 'password': '', 'user_id': '', 'nickname': ''}
        self.session.cookies.save()
        self.storage.save()

    def _raw_request(self, method: str, endpoint: str, data=None) -> Any:
        resp = None
        if method == 'GET':
            resp = self.session.get(endpoint, params=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        elif method == 'POST':
            resp = self.session.post(endpoint, data=data, headers=self.header, timeout=DEFAULT_TIMEOUT)
        return resp

    def make_cookie(self, name: str, value: str) -> Cookie:
        return Cookie(version=0, name=name, value=value, port=None, port_specified=False, domain='music.163.com', domain_specified=True, domain_initial_dot=False, path='/', path_specified=True, secure=False, expires=None, discard=False, comment=None, comment_url=None, rest={})

    def request(self, method: str, path: str, params={}, default={'code': -1}, custom_cookies={}) -> Any:
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

    def login(self, username: str, password: str) -> Dict[str, Any]:
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

    def daily_task(self, is_mobile: bool = True) -> Any:
        path = '/weapi/point/dailyTask'
        params = dict(type=0 if is_mobile else 1)
        return self.request('POST', path, params)

    def user_playlist(self, uid: str, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        path = '/weapi/user/playlist'
        params = dict(uid=uid, offset=offset, limit=limit)
        return self.request('POST', path, params).get('playlist', [])

    def recommend_resource(self) -> List[Dict[str, Any]]:
        path = '/weapi/v1/discovery/recommend/resource'
        return self.request('POST', path).get('recommend', [])

    def recommend_playlist(self, total: bool = True, offset: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        path = '/weapi/v1/discovery/recommend/songs'
        params = dict(total=total, offset=offset, limit=limit)
        return self.request('POST', path, params).get('recommend', [])

    # Add type annotations for the remaining methods similarly
