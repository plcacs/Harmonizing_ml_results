from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast

class Address(typing.NamedTuple):
    pass

_KeyType = typing.TypeVar('_KeyType')
_CovariantValueType = typing.TypeVar('_CovariantValueType', covariant=True)

class URL:
    def __init__(self, url: str = '', scope: Optional[Scope] = None, **components: Any) -> None:
        if scope is not None:
            assert not url, 'Cannot set both "url" and "scope".'
            assert not components, 'Cannot set both "scope" and "**components".'
            scheme = scope.get('scheme', 'http')
            server = scope.get('server', None)
            path = scope['path']
            query_string = scope.get('query_string', b'')
            host_header = None
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
                default_port = {'http': 80, 'https': 443, 'ws': 80, 'wss': 443}[scheme]
                if port == default_port:
                    url = f'{scheme}://{host}{path}'
                else:
                    url = f'{scheme}://{host}:{port}{path}'
            if query_string:
                url += '?' + query_string.decode()
        elif components:
            assert not url, 'Cannot set both "url" and "**components".'
            url = URL('').replace(**components).components.geturl()
        self._url = url

    @property
    def components(self) -> SplitResult:
        if not hasattr(self, '_components'):
            self._components = urlsplit(self._url)
        return self._components

    @property
    def scheme(self) -> str:
        return self.components.scheme

    @property
    def netloc(self) -> str:
        return self.components.netloc

    @property
    def path(self) -> str:
        return self.components.path

    @property
    def query(self) -> str:
        return self.components.query

    @property
    def fragment(self) -> str:
        return self.components.fragment

    @property
    def username(self) -> Optional[str]:
        return self.components.username

    @property
    def password(self) -> Optional[str]:
        return self.components.password

    @property
    def hostname(self) -> Optional[str]:
        return self.components.hostname

    @property
    def port(self) -> Optional[int]:
        return self.components.port

    @property
    def is_secure(self) -> bool:
        return self.scheme in ('https', 'wss')

    def replace(self, **kwargs: Any) -> URL:
        if 'username' in kwargs or 'password' in kwargs or 'hostname' in kwargs or ('port' in kwargs):
            hostname = kwargs.pop('hostname', None)
            port = kwargs.pop('port', self.port)
            username = kwargs.pop('username', self.username)
            password = kwargs.pop('password', self.password)
            if hostname is None:
                netloc = self.netloc
                _, _, hostname = netloc.rpartition('@')
                if hostname[-1] != ']':
                    hostname = hostname.rsplit(':', 1)[0]
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

    def include_query_params(self, **kwargs: Any) -> URL:
        params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
        params.update({str(key): str(value) for key, value in kwargs.items()})
        query = urlencode(params.multi_items())
        return self.replace(query=query)

    def replace_query_params(self, **kwargs: Any) -> URL:
        query = urlencode([(str(key), str(value)) for key, value in kwargs.items()])
        return self.replace(query=query)

    def remove_query_params(self, keys: Union[str, List[str]]) -> URL:
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
    def __new__(cls, path: str, protocol: str = '', host: str = '') -> URLPath:
        assert protocol in ('http', 'websocket', '')
        return str.__new__(cls, path)

    def __init__(self, path: str, protocol: str = '', host: str = '') -> None:
        self.protocol = protocol
        self.host = host

    def make_absolute_url(self, base_url: Union[str, URL]) -> URL:
        if isinstance(base_url, str):
            base_url = URL(base_url)
        if self.protocol:
            scheme = {'http': {True: 'https', False: 'http'}, 'websocket': {True: 'wss', False: 'ws'}}[self.protocol][base_url.is_secure]
        else:
            scheme = base_url.scheme
        netloc = self.host or base_url.netloc
        path = base_url.path.rstrip('/') + str(self)
        return URL(scheme=scheme, netloc=netloc, path=path)

class Secret:
    def __init__(self, value: str) -> None:
        self._value = value

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
            self._items = [item.strip() for item in splitter]
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
        return ', '.join((repr(item) for item in self))

class ImmutableMultiDict(typing.Mapping[_KeyType, _CovariantValueType]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert len(args) < 2, 'Too many arguments.'
        value = args[0] if args else []
        if kwargs:
            value = ImmutableMultiDict(value).multi_items() + ImmutableMultiDict(kwargs).multi_items()
        if not value:
            _items: List[Tuple[Any, Any]] = []
        elif hasattr(value, 'multi_items'):
            value = typing.cast(ImmutableMultiDict[_KeyType, _CovariantValueType], value)
            _items = list(value.multi_items())
        elif hasattr(value, 'items'):
            value = typing.cast(typing.Mapping[_KeyType, _CovariantValueType], value)
            _items = list(value.items())
        else:
            value = typing.cast('list[tuple[typing.Any, typing.Any]]', value)
            _items = list(value)
        self._dict = {k: v for k, v in _items}
        self._list = _items

    def getlist(self, key: _KeyType) -> List[_CovariantValueType]:
        return [item_value for item_key, item_value in self._list if item_key == key]

    def keys(self) -> typing.KeysView[_KeyType]:
        return self._dict.keys()

    def values(self) -> typing.ValuesView[_CovariantValueType]:
        return self._dict.values()

    def items(self) -> typing.ItemsView[_KeyType, _CovariantValueType]:
        return self._dict.items()

    def multi_items(self) -> List[Tuple[_KeyType, _CovariantValueType]]:
        return list(self._list)

    def __getitem__(self, key: _KeyType) -> _CovariantValueType:
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
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

class MultiDict(ImmutableMultiDict[typing.Any, typing.Any]):
    def __setitem__(self, key: str, value: Any) -> None:
        self.setlist(key, [value])

    def __delitem__(self, key: str) -> None:
        self._list = [(k, v) for k, v in self._list if k != key]
        del self._dict[key]

    def pop(self, key: str, default: Any = None) -> Any:
        self._list = [(k, v) for k, v in self._list if k != key]
        return self._dict.pop(key, default)

    def popitem(self) -> Tuple[str, Any]:
        key, value = self._dict.popitem()
        self._list = [(k, v) for k, v in self._list if k != key]
        return (key, value)

    def poplist(self, key: str) -> List[Any]:
        values = [v for k, v in self._list if k == key]
        self.pop(key)
        return values

    def clear(self) -> None:
        self._dict.clear()
        self._list.clear()

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self._dict[key] = default
            self._list.append((key, default))
        return self[key]

    def setlist(self, key: str, values: List[Any]) -> None:
        if not values:
            self.pop(key, None)
        else:
            existing_items = [(k, v) for k, v in self._list if k != key]
            self._list = existing_items + [(key, value) for value in values]
            self._dict[key] = values[-1]

    def append(self, key: str, value: Any) -> None:
        self._list.append((key, value))
        self._dict[key] = value

    def update(self, *args: Any, **kwargs: Any) -> None:
        value = MultiDict(*args, **kwargs)
        existing_items = [(k, v) for k, v in self._list if k not in value.keys()]
        self._list = existing_items + value.multi_items()
        self._dict.update(value)

class QueryParams(ImmutableMultiDict[str, str]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert len(args) < 2, 'Too many arguments.'
        value = args[0] if args else []
        if isinstance(value, str):
            super().__init__(parse_qsl(value, keep_blank_values=True), **kwargs)
        elif isinstance(value, bytes):
            super().__init__(parse_qsl(value.decode('latin-1'), keep_blank_values=True), **kwargs)
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
    def __init__(
        self,
        file: typing.IO[bytes],
        *,
        size: Optional[int] = None,
        filename: Optional[str] = None,
        headers: Optional[Headers] = None
    ) -> None:
        self.filename = filename
        self.file = file
        self.size = size
        self.headers = headers or Headers()

    @property
    def content_type(self) -> Optional[str]:
        return self.headers.get('content-type', None)

    @property
    def _in_memory(self) -> bool:
        rolled_to_disk = getattr(self.file, '_rolled', True)
        return not rolled_to_disk

    async def write(self, data: bytes) -> None:
        if self.size is not None:
            self.size += len(data)
        if self._in_memory:
            self.file.write(data)
        else:
            await run_in_threadpool(self.file.write, data)

    async def read(self, size: int = -1) -> bytes:
        if self._in_memory:
            return self.file.read(size)
        return await run_in_threadpool(self.file.read, size)

    async def seek(self, offset: int) -> None:
        if self._in_memory:
            self.file.seek(offset)
        else:
            await run_in_threadpool(self.file.seek, offset)

    async def close(self) -> None:
        if self._in_memory:
            self.file.close()
        else:
            await run_in_threadpool(self.file.close)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(filename={self.filename!r}, size={self.size!r}, headers={self.headers!r})'

class FormData(ImmutableMultiDict[str, Union[UploadFile, str]]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def close(self) -> None:
        for key, value in self.multi_items():
            if isinstance(value, UploadFile):
                await value.close()

class Headers(typing.Mapping[str, str]):
    def __init__(
        self,
        headers: Optional[Mapping[str, str]] = None,
        raw: Optional[List[Tuple[bytes, bytes]]] = None,
        scope: Optional[Scope] = None
    ) -> None:
        self._list: List[Tuple[bytes, bytes]] = []
        if headers is not None:
            assert raw is None, 'Cannot set both "headers" and "raw".'
            assert scope is None, 'Cannot set both "headers" and "scope".'
            self._list = [(key.lower().encode('latin-1'), value.encode('latin-1')) for key, value in headers.items()]
        elif raw is not None:
            assert scope is None, 'Cannot set both "raw" and "scope".'
            self._list = raw
        elif scope is not None:
            self._list = scope['headers'] = list(scope['headers'])

    @property
    def raw(self) -> List[Tuple[bytes, bytes]]:
        return list(self._list)

    def keys(self) -> List[str]:
        return [key.decode('latin-1') for key, value in self._list]

    def values(self) -> List[str]:
        return [value.decode('latin-1') for key, value in self._list]

    def items(self) -> List[Tuple[str, str]]:
        return [(key.decode('latin-1'), value.decode('latin-1')) for key, value in self._list]

    def getlist(self, key: str) -> List[str]:
        get_header_key = key.lower().encode('latin-1')
        return [item_value.decode('latin-1') for item_key, item_value in self._list if item_key == get_header_key]

    def mutablecopy(self) -> MutableHeaders:
        return MutableHeaders(raw=self._list[:])

    def __getitem__(self, key: str) -> str:
        get_header_key = key.lower().encode('latin-1')
        for header_key, header_value in self._list:
            if header_key == get_header_key:
                return header_value.decode('latin-1')
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        get_header_key = key.lower().encode('latin-1')
        for header_key, header