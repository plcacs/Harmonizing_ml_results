from __future__ import annotations
import errno
import importlib.util
import os
import stat
import typing
from email.utils import parsedate
import anyio
import anyio.to_thread
from starlette._utils import get_route_path
from starlette.datastructures import URL, Headers
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.types import Receive, Scope, Send

PathLike = typing.Union[str, os.PathLike[str]]
PackageType = typing.Union[str, typing.Tuple[str, str]]

class NotModifiedResponse(Response):
    NOT_MODIFIED_HEADERS = ('cache-control', 'content-location', 'date', 'etag', 'expires', 'vary')

    def __init__(self, headers: typing.Mapping[str, str]) -> None:
        filtered_headers = {name: value for name, value in headers.items() if name in self.NOT_MODIFIED_HEADERS}
        super().__init__(status_code=304, headers=filtered_headers)

class StaticFiles:

    def __init__(
        self,
        *,
        directory: typing.Optional[PathLike] = None,
        packages: typing.Optional[typing.Sequence[PackageType]] = None,
        html: bool = False,
        check_dir: bool = True,
        follow_symlink: bool = False
    ) -> None:
        self.directory: typing.Optional[PathLike] = directory
        self.packages: typing.Optional[typing.Sequence[PackageType]] = packages
        self.all_directories: typing.List[str] = self.get_directories(directory, packages)
        self.html: bool = html
        self.config_checked: bool = False
        self.follow_symlink: bool = follow_symlink
        if check_dir and directory is not None and (not os.path.isdir(str(directory))):
            raise RuntimeError(f"Directory '{directory}' does not exist")

    def get_directories(
        self,
        directory: typing.Optional[PathLike] = None,
        packages: typing.Optional[typing.Sequence[PackageType]] = None
    ) -> typing.List[str]:
        """
        Given `directory` and `packages` arguments, return a list of all the
        directories that should be used for serving static files from.
        """
        directories: typing.List[str] = []
        if directory is not None:
            directories.append(str(directory))
        for package in packages or []:
            if isinstance(package, tuple):
                package_name, statics_dir = package
            else:
                package_name = package
                statics_dir = 'statics'
            spec = importlib.util.find_spec(package_name)
            assert spec is not None, f'Package {package_name!r} could not be found.'
            assert spec.origin is not None, f'Package {package_name!r} could not be found.'
            package_directory = os.path.normpath(os.path.join(spec.origin, '..', statics_dir))
            assert os.path.isdir(package_directory), f"Directory '{statics_dir!r}' in package {package_name!r} could not be found."
            directories.append(package_directory)
        return directories

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        The ASGI entry point.
        """
        assert scope['type'] == 'http'
        if not self.config_checked:
            await self.check_config()
            self.config_checked = True
        path: str = self.get_path(scope)
        response: Response = await self.get_response(path, scope)
        await response(scope, receive, send)

    def get_path(self, scope: Scope) -> str:
        """
        Given the ASGI scope, return the `path` string to serve up,
        with OS specific path separators, and any '..', '.' components removed.
        """
        route_path: str = get_route_path(scope)
        return os.path.normpath(os.path.join(*route_path.split('/')))

    async def get_response(self, path: str, scope: Scope) -> Response:
        """
        Returns an HTTP response, given the incoming path, method and request headers.
        """
        if scope['method'] not in ('GET', 'HEAD'):
            raise HTTPException(status_code=405)
        try:
            full_path, stat_result = await anyio.to_thread.run_sync(self.lookup_path, path)
        except PermissionError:
            raise HTTPException(status_code=401)
        except OSError as exc:
            if exc.errno == errno.ENAMETOOLONG:
                raise HTTPException(status_code=404)
            raise exc
        if stat_result is not None and stat.S_ISREG(stat_result.st_mode):
            return self.file_response(full_path, stat_result, scope)
        elif stat_result is not None and stat.S_ISDIR(stat_result.st_mode) and self.html:
            index_path: str = os.path.join(path, 'index.html')
            full_path, stat_result = await anyio.to_thread.run_sync(self.lookup_path, index_path)
            if stat_result is not None and stat.S_ISREG(stat_result.st_mode):
                if not scope['path'].endswith('/'):
                    url: URL = URL(scope=scope)
                    url = url.replace(path=url.path + '/')
                    return RedirectResponse(url=url)
                return self.file_response(full_path, stat_result, scope)
        if self.html:
            full_path, stat_result = await anyio.to_thread.run_sync(self.lookup_path, '404.html')
            if stat_result is not None and stat.S_ISREG(stat_result.st_mode):
                return FileResponse(full_path, stat_result=stat_result, status_code=404)
        raise HTTPException(status_code=404)

    def lookup_path(self, path: str) -> typing.Tuple[str, typing.Optional[os.stat_result]]:
        for directory in self.all_directories:
            joined_path: str = os.path.join(directory, path)
            if self.follow_symlink:
                full_path: str = os.path.abspath(joined_path)
            else:
                full_path = os.path.realpath(joined_path)
                directory = os.path.realpath(directory)
            if os.path.commonpath([full_path, directory]) != str(directory):
                continue
            try:
                return (full_path, os.stat(full_path))
            except (FileNotFoundError, NotADirectoryError):
                continue
        return ('', None)

    def file_response(self, full_path: str, stat_result: os.stat_result, scope: Scope, status_code: int = 200) -> Response:
        request_headers: Headers = Headers(scope=scope)
        response: FileResponse = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
        if self.is_not_modified(response.headers, request_headers):
            return NotModifiedResponse(response.headers)
        return response

    async def check_config(self) -> None:
        """
        Perform a one-off configuration check that StaticFiles is actually
        pointed at a directory, so that we can raise loud errors rather than
        just returning 404 responses.
        """
        if self.directory is None:
            return
        try:
            stat_result: os.stat_result = await anyio.to_thread.run_sync(os.stat, self.directory)
        except FileNotFoundError:
            raise RuntimeError(f"StaticFiles directory '{self.directory}' does not exist.")
        if not (stat.S_ISDIR(stat_result.st_mode) or stat.S_ISLNK(stat_result.st_mode)):
            raise RuntimeError(f"StaticFiles path '{self.directory}' is not a directory.")

    def is_not_modified(self, response_headers: Headers, request_headers: Headers) -> bool:
        """
        Given the request and response headers, return `True` if an HTTP
        "Not Modified" response could be returned instead.
        """
        try:
            if_none_match: str = request_headers['if-none-match']
            etag: str = response_headers['etag']
            if etag in [tag.strip(' W/') for tag in if_none_match.split(',')]:
                return True
        except KeyError:
            pass
        try:
            if_modified_since_tuple = parsedate(request_headers['if-modified-since'])
            last_modified_tuple = parsedate(response_headers['last-modified'])
            if if_modified_since_tuple is not None and last_modified_tuple is not None and (if_modified_since_tuple >= last_modified_tuple):
                return True
        except KeyError:
            pass
        return False