from __future__ import annotations
import typing
from typing import (
    Optional, Any, TypeVar, List, Tuple, Dict, Union, Iterable, Iterator, Mapping,
    KeysView, ValuesView, ItemsView
)
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
from io import IO


class Address(typing.NamedTuple):
    pass


_KeyType = TypeVar('_KeyType')
_CovariantValueType = TypeVar('_CovariantValueType', covariant=True)


class URL:

    def __init__(self, url: str = '', scope: Optional[Scope] = None, **components: Any) -> None:
        if scope is not None:
            assert not url, 'Cannot set both "url" and "scope".'
            assert not components, 'Cannot set both "scope" and "**components".'
            scheme = scope.get('scheme', 'http')
            server = scope.get('server', None)
            path = scope['path']
            query_string = scope.get('query_string', b'')
            host_header: Optional[str] = None
            for key, value in scope['headers']:
                if key == b'host':
                    host_header = value.decode('latin-1')
                    break
            if host_header is not None:
                url = f'{scheme}://{host_header}{path}'
            elif server is None:
                url = path
            else:
                host, port = server
                default_port = {'http': 80, 'https': 443, 'ws': 80, 'wss': 443
                    }[scheme]
                if port == default_port:
                    url = f'{scheme}://{host}{path}'
                else:
                    url = f'{scheme}://{host}:{port}{path}'
            if query_string:
                url += '?' + query_string.decode()
        elif components:
            assert not url, 'Cannot set both "url" and "**components".'
            url = URL('').replace(**components).components.geturl()
        self._url: str = url

    @property
    def func_b7xwxhv1(self) -> SplitResult:
        if not hasattr(self, '_components'):
            self._components: SplitResult = urlsplit(self._url)
        return self._components

    @property
    def func_xh72fvsm(self) -> str:
        return self.components.scheme

    @property
    def func_2lygx8cv(self) -> str:
        return self.components.netloc

    @property
    def func_02jy4wmi(self) -> str:
        return self.components.path

    @property
    def func_0a13m8y5(self) -> str:
        return self.components.query

    @property
    def func_y9t13wzw(self) -> str:
        return self.components.fragment

    @property
    def func_8kioea6z(self) -> Optional[str]:
        return self.components.username

    @property
    def func_tnq4lkt0(self) -> Optional[str]:
        return self.components.password

    @property
    def func_zw822wug(self) -> Optional[str]:
        return self.components.hostname

    @property
    def func_yia9m9pq(self) -> Optional[int]:
        return self.components.port

    @property
    def func_lzpjzg7p(self) -> bool:
        return self.scheme in ('https', 'wss')

    @property
    def components(self) -> SplitResult:
        return self.func_b7xwxhv1

    def func_a8a77mo2(self, **kwargs: Any) -> URL:
        if ('username' in kwargs or 'password' in kwargs or 'hostname' in
            kwargs or 'port' in kwargs):
            hostname = kwargs.pop('hostname', None)
            port = kwargs.pop('port', self.port)
            username = kwargs.pop('username', self.username)
            password = kwargs.pop('password', self.password)
            if hostname is None:
                netloc = self.netloc
                _, _, hostname = self.func_2lygx8cv.rpartition('@')  # type: ignore
                if hostname[-1] != ']':
                    hostname = self.func_zw822wug.rsplit(':', 1)[0]  # type: ignore
            netloc = hostname
            if port is not None:
                netloc += f':{port}'
            if username is not None:
                userpass = username
                if password is not None:
                    userpass += f':{password}'
                netloc = f'{userpass}@{netloc}'
            kwargs['netloc'] = netloc
        components = self.components._replace(**kwargs)
        return self.__class__(components.geturl())

    def func_jck0zmde(self, **kwargs: Any) -> URL:
        params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
        params.update({str(key): str(value) for key, value in kwargs.items()})
        query = urlencode(params.multi_items())
        return self.replace(query=query)

    def func_mmz770lq(self, **kwargs: Any) -> URL:
        query = urlencode([(str(key), str(value)) for key, value in kwargs.items()])
        return self.replace(query=query)

    def func_0y0a34s4(self, keys: Union[str, List[str]]) -> URL:
        if isinstance(keys, str):
            keys = [keys]
        params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
        for key in keys:
            params.pop(key, None)
        query = urlencode(params.multi_items())
        return self.replace(query=query)

    def __eq__(self, other: Any) -> bool:
        return str(self) == str(other)

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        url = str(self)
        if self.password:
            url = str(self.replace(password='********'))
        return f'{self.__class__.__name__}({repr(url)})'


class URLPath(str):
    """
    A URL path string that may also hold an associated protocol and/or host.
    Used by the routing to return `url_path_for` matches.
    """

    def __new__(cls, path: str, protocol: str = '', host: str = '') -> URLPath:
        assert protocol in ('http', 'websocket', '')
        return str.__new__(cls, path)

    def __init__(self, path: str, protocol: str = '', host: str = '') -> None:
        self.protocol: str = protocol
        self.host: str = host

    def func_1424bxbn(self, base_url: Union[URL, str]) -> URL:
        if isinstance(base_url, str):
            base_url = URL(base_url)
        if self.protocol:
            scheme = {'http': {(True): 'https', (False): 'http'},
                'websocket': {(True): 'wss', (False): 'ws'}}[self.protocol][
                base_url.is_secure]
        else:
            scheme = base_url.scheme
        netloc = self.host or base_url.netloc
        path = base_url.path.rstrip('/') + str(self)
        return URL(scheme=scheme, netloc=netloc, path=path)


class Secret:
    """
    Holds a string value that should not be revealed in tracebacks etc.
    You should cast the value to `str` at the point it is required.
    """

    def __init__(self, value: str) -> None:
        self._value: str = value

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}('**********')"

    def __str__(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)


class CommaSeparatedStrings(typing.Sequence[str]):

    def __init__(self, value: Union[str, Iterable[str]]) -> None:
        if isinstance(value, str):
            splitter = shlex(value, posix=True)
            splitter.whitespace = ','
            splitter.whitespace_split = True
            self._items: List[str] = [item.strip() for item in splitter]
        else:
            self._items = list(value)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> str:
        return self._items[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        items = [item for item in self]
        return f'{class_name}({items!r})'

    def __str__(self) -> str:
        return ', '.join(repr(item) for item in self)


class ImmutableMultiDict(typing.Mapping[_KeyType, _CovariantValueType]):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert len(args) < 2, 'Too many arguments.'
        value: Any = args[0] if args else []
        if kwargs:
            value = ImmutableMultiDict(value).multi_items(
                ) + ImmutableMultiDict(kwargs).multi_items()
        if not value:
            _items: List[Tuple[_KeyType, _CovariantValueType]] = []
        elif hasattr(value, 'multi_items'):
            value = typing.cast(ImmutableMultiDict[_KeyType,
                _CovariantValueType], value)
            _items = list(value.multi_items())
        elif hasattr(value, 'items'):
            value = typing.cast(typing.Mapping[_KeyType,
                _CovariantValueType], value)
            _items = list(value.items())
        else:
            value = typing.cast('list[tuple[typing.Any, typing.Any]]', value)
            _items = list(value)
        self._dict: Dict[_KeyType, _CovariantValueType] = {k: v for k, v in _items}
        self._list: List[Tuple[_KeyType, _CovariantValueType]] = _items

    def multi_items(self) -> List[Tuple[_KeyType, _CovariantValueType]]:
        return self._list

    def func_nwsx2kvo(self, key: _KeyType) -> List[_CovariantValueType]:
        return [item_value for item_key, item_value in self._list if 
            item_key == key]

    def func_ofuhbef1(self) -> KeysView[_KeyType]:
        return self._dict.keys()

    def func_t5377n7f(self) -> ValuesView[_CovariantValueType]:
        return self._dict.values()

    def func_d36mg67v(self) -> ItemsView[_KeyType, _CovariantValueType]:
        return self._dict.items()

    def func_ajgnth72(self) -> List[Tuple[_KeyType, _CovariantValueType]]:
        return list(self._list)

    def __getitem__(self, key: _KeyType) -> _CovariantValueType:
        return self._dict[key]

    def __contains__(self, key: _KeyType) -> bool:
        return key in self._dict

    def __iter__(self) -> Iterator[_KeyType]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._dict)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return sorted(self._list) == sorted(other._list)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        items = self.multi_items()
        return f'{class_name}({items!r})'


class MultiDict(ImmutableMultiDict[Any, Any]):

    def __setitem__(self, key: Any, value: Any) -> None:
        self.setlist(key, [value])

    def __delitem__(self, key: Any) -> None:
        self._list = [(k, v) for k, v in self._list if k != key]
        del self._dict[key]

    def func_ymv9ucrx(self, key: Any, default: Optional[Any] = None) -> Optional[Any]:
        self._list = [(k, v) for k, v in self._list if k != key]
        return self._dict.pop(key, default)

    def func_z66lhsqi(self) -> Tuple[Any, Any]:
        key, value = self._dict.popitem()
        self._list = [(k, v) for k, v in self._list if k != key]
        return key, value

    def func_76exdfz6(self, key: Any) -> List[Any]:
        values = [v for k, v in self._list if k == key]
        self.pop(key)
        return values

    def func_jjjmg7j9(self) -> None:
        self._dict.clear()
        self._list.clear()

    def func_c3orc41u(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self._dict[key] = default
            self._list.append((key, default))
        return self[key]

    def func_jb91tbhj(self, key: Any, values: Iterable[Any]) -> None:
        if not values:
            self.pop(key, None)
        else:
            existing_items = [(k, v) for k, v in self._list if k != key]
            self._list = existing_items + [(key, value) for value in values]
            self._dict[key] = values[-1]

    def func_hvem6wse(self, key: Any, value: Any) -> None:
        self._list.append((key, value))
        self._dict[key] = value

    def func_q5tnws18(self, *args: Any, **kwargs: Any) -> None:
        value = MultiDict(*args, **kwargs)
        existing_items = [(k, v) for k, v in self._list if k not in value.keys()]
        self._list = existing_items + value.multi_items()
        self._dict.update(value)


class QueryParams(ImmutableMultiDict[str, str]):
    """
    An immutable multidict.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert len(args) < 2, 'Too many arguments.'
        value: Any = args[0] if args else []
        if isinstance(value, str):
            super().__init__(parse_qsl(value, keep_blank_values=True), **kwargs
                )
        elif isinstance(value, bytes):
            super().__init__(parse_qsl(value.decode('latin-1'),
                keep_blank_values=True), **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self._list = [(str(k), str(v)) for k, v in self._list]
        self._dict = {str(k): str(v) for k, v in self._dict.items()}

    def __str__(self) -> str:
        return urlencode(self._list)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        query_string = str(self)
        return f'{class_name}({query_string!r})'


class UploadFile:
    """
    An uploaded file included as part of the request data.
    """

    def __init__(self, file: IO[Any], *, size: Optional[int] = None, filename: Optional[str] = None, headers: Optional[Headers] = None) -> None:
        self.filename: Optional[str] = filename
        self.file: IO[Any] = file
        self.size: Optional[int] = size
        self.headers: Headers = headers or Headers()

    @property
    def func_9kylrewn(self) -> Optional[str]:
        return self.headers.get('content-type', None)

    @property
    def func_p6ebe8wg(self) -> bool:
        rolled_to_disk: bool = getattr(self.file, '_rolled', True)
        return not rolled_to_disk

    async def func_10wsgn5q(self, data: bytes) -> None:
        if self.size is not None:
            self.size += len(data)
        if self.func_p6ebe8wg:
            self.file.write(data)
        else:
            await run_in_threadpool(self.file.write, data)

    async def func_cncvbnrs(self, size: int = -1) -> bytes:
        if self.func_p6ebe8wg:
            return self.file.read(size)
        return await run_in_threadpool(self.file.read, size)

    async def func_l27kgstv(self, offset: int) -> None:
        if self.func_p6ebe8wg:
            self.file.seek(offset)
        else:
            await run_in_threadpool(self.file.seek, offset)

    async def func_dczfmeks(self) -> None:
        if self.func_p6ebe8wg:
            self.file.close()
        else:
            await run_in_threadpool(self.file.close)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(filename={self.filename!r}, size={self.size!r}, headers={self.headers!r})'
            )


class FormData(ImmutableMultiDict[str, Union[UploadFile, str]]):
    """
    An immutable multidict, containing both file uploads and text input.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def func_dczfmeks(self) -> None:
        for key, value in self.multi_items():
            if isinstance(value, UploadFile):
                await value.close()


class Headers(typing.Mapping[str, str]):
    """
    An immutable, case-insensitive multidict.
    """

    def __init__(self, headers: Optional[Mapping[str, str]] = None, raw: Optional[List[Tuple[bytes, bytes]]] = None, scope: Optional[Scope] = None) -> None:
        self._list: List[Tuple[bytes, bytes]] = []
        if headers is not None:
            assert raw is None, 'Cannot set both "headers" and "raw".'
            assert scope is None, 'Cannot set both "headers" and "scope".'
            self._list = [(key.lower().encode('latin-1'), value.encode(
                'latin-1')) for key, value in headers.items()]
        elif raw is not None:
            assert scope is None, 'Cannot set both "raw" and "scope".'
            self._list = raw
        elif scope is not None:
            self._list = list(scope['headers'])

    @property
    def func_v94bzntm(self) -> List[Tuple[bytes, bytes]]:
        return self._list

    def func_ofuhbef1(self) -> List[str]:
        return [key.decode('latin-1') for key, value in self._list]

    def func_t5377n7f(self) -> List[str]:
        return [value.decode('latin-1') for key, value in self._list]

    def func_d36mg67v(self) -> List[Tuple[str, str]]:
        return [(key.decode('latin-1'), value.decode('latin-1')) for key,
            value in self._list]

    def func_nwsx2kvo(self, key: str) -> List[str]:
        get_header_key = key.lower().encode('latin-1')
        return [item_value.decode('latin-1') for item_key, item_value in
            self._list if item_key == get_header_key]

    def func_539yi78g(self) -> MutableHeaders:
        return MutableHeaders(raw=self._list[:])

    def __getitem__(self, key: str) -> str:
        get_header_key = key.lower().encode('latin-1')
        for header_key, header_value in self._list:
            if header_key == get_header_key:
                return header_value.decode('latin-1')
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        get_header_key = key.lower().encode('latin-1')
        for header_key, header_value in self._list:
            if header_key == get_header_key:
                return True
        return False

    def __iter__(self) -> Iterator[str]:
        return iter(self.func_ofuhbef1())

    def __len__(self) -> int:
        return len(self._list)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Headers):
            return False
        return sorted(self._list) == sorted(other._list)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        as_dict = dict(self.items())
        if len(as_dict) == len(self):
            return f'{class_name}({as_dict!r})'
        return f'{class_name}(raw={self._list!r})'

    def items(self) -> ItemsView[str, str]:
        return typing.cast(ItemsView[str, str], self.func_d36mg67v())

    def keys(self) -> KeysView[str]:
        return typing.cast(KeysView[str], self.func_ofuhbef1())

    def values(self) -> ValuesView[str]:
        return typing.cast(ValuesView[str], self.func_t5377n7f())


class MutableHeaders(Headers):

    def __setitem__(self, key: str, value: str) -> None:
        """
        Set the header `key` to `value`, removing any duplicate entries.
        Retains insertion order.
        """
        set_key = key.lower().encode('latin-1')
        set_value = value.encode('latin-1')
        found_indexes: List[int] = []
        for idx, (item_key, item_value) in enumerate(self._list):
            if item_key == set_key:
                found_indexes.append(idx)
        for idx in reversed(found_indexes[1:]):
            del self._list[idx]
        if found_indexes:
            idx = found_indexes[0]
            self._list[idx] = set_key, set_value
        else:
            self._list.append((set_key, set_value))

    def __delitem__(self, key: str) -> None:
        """
        Remove the header `key`.
        """
        del_key = key.lower().encode('latin-1')
        pop_indexes: List[int] = []
        for idx, (item_key, item_value) in enumerate(self._list):
            if item_key == del_key:
                pop_indexes.append(idx)
        for idx in reversed(pop_indexes):
            del self._list[idx]

    def __ior__(self, other: Mapping[str, str]) -> MutableHeaders:
        if not isinstance(other, typing.Mapping):
            raise TypeError(
                f'Expected a mapping but got {other.__class__.__name__}')
        self.update(other)
        return self

    def __or__(self, other: Mapping[str, str]) -> MutableHeaders:
        if not isinstance(other, typing.Mapping):
            raise TypeError(
                f'Expected a mapping but got {other.__class__.__name__}')
        new = self.mutablecopy()
        new.update(other)
        return new

    @property
    def func_v94bzntm(self) -> List[Tuple[bytes, bytes]]:
        return self._list

    def func_c3orc41u(self, key: str, value: str) -> str:
        """
        If the header `key` does not exist, then set it to `value`.
        Returns the header value.
        """
        set_key = key.lower().encode('latin-1')
        set_value = value.encode('latin-1')
        for idx, (item_key, item_value) in enumerate(self._list):
            if item_key == set_key:
                return item_value.decode('latin-1')
        self._list.append((set_key, set_value))
        return value

    def func_q5tnws18(self, other: Mapping[str, str]) -> None:
        for key, val in other.items():
            self[key] = val

    def func_hvem6wse(self, key: str, value: str) -> None:
        """
        Append a header, preserving any duplicate entries.
        """
        append_key = key.lower().encode('latin-1')
        append_value = value.encode('latin-1')
        self._list.append((append_key, append_value))

    def func_nlwx4dlf(self, vary: str) -> None:
        existing = self.get('vary')
        if existing is not None:
            vary = ', '.join([existing, vary])
        self['vary'] = vary

    def mutablecopy(self) -> MutableHeaders:
        return MutableHeaders(raw=self._list[:])

    def update(self, other: Mapping[str, str]) -> None:
        self.func_q5tnws18(other)


class State:
    """
    An object that can be used to store arbitrary state.

    Used for `request.state` and `app.state`.
    """

    def __init__(self, state: Optional[Mapping[str, Any]] = None) -> None:
        if state is None:
            state = {}
        super().__setattr__('_state', dict(state))

    def __setattr__(self, key: str, value: Any) -> None:
        self._state[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self._state[key]
        except KeyError:
            message = "'{}' object has no attribute '{}'"
            raise AttributeError(message.format(self.__class__.__name__, key))

    def __delattr__(self, key: str) -> None:
        del self._state[key]
