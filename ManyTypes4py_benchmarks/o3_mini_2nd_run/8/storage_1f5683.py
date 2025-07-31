from __future__ import annotations
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, TypedDict, Union, runtime_checkable, List
from urllib.parse import urlparse, urlsplit, urlunparse, ParseResult
from uuid import uuid4

import fsspec
from anyio import run_process
from pydantic import SecretStr
from prefect._internal.concurrency.api import create_call, from_async
from prefect.blocks.core import Block, BlockNotSavedError
from prefect.blocks.system import Secret
from prefect.filesystems import ReadableDeploymentStorage, WritableDeploymentStorage
from prefect.logging.loggers import get_logger
from prefect.utilities.collections import visit_collection

@runtime_checkable
class RunnerStorage(Protocol):
    def set_base_path(self, path: Union[str, Path]) -> None:
        ...

    @property
    def pull_interval(self) -> Optional[int]:
        ...

    @property
    def destination(self) -> Path:
        ...

    async def pull_code(self) -> None:
        ...

    def to_pull_step(self) -> Dict[str, Any]:
        ...

    def __eq__(self, __value: Any) -> bool:
        ...

class GitCredentials(TypedDict, total=False):
    ...

class GitRepository:
    def __init__(
        self,
        url: str,
        credentials: Optional[Union[Dict[str, Any], Block]] = None,
        name: Optional[str] = None,
        branch: Optional[str] = None,
        include_submodules: bool = False,
        pull_interval: Optional[int] = 60,
        directories: Optional[List[str]] = None,
    ) -> None:
        if credentials is None:
            credentials = {}
        if isinstance(credentials, dict) and credentials.get('username') and not (credentials.get('access_token') or credentials.get('password')):
            raise ValueError('If a username is provided, an access token or password must also be provided.')
        self._url: str = url
        self._branch: Optional[str] = branch
        self._credentials: Union[Dict[str, Any], Block] = credentials
        self._include_submodules: bool = include_submodules
        repo_name = urlparse(url).path.split('/')[-1].replace('.git', '')
        default_name = f'{repo_name}-{branch}' if branch else repo_name
        self._name: str = name or default_name
        self._logger = get_logger(f'runner.storage.git-repository.{self._name}')
        self._storage_base_path: Path = Path.cwd()
        self._pull_interval: Optional[int] = pull_interval
        self._directories: Optional[List[str]] = directories

    @property
    def destination(self) -> Path:
        return self._storage_base_path / self._name

    def set_base_path(self, path: Union[str, Path]) -> None:
        self._storage_base_path = Path(path)

    @property
    def pull_interval(self) -> Optional[int]:
        return self._pull_interval

    @property
    def _formatted_credentials(self) -> Optional[str]:
        if not self._credentials:
            return None
        credentials: Dict[str, Any]
        if isinstance(self._credentials, Block):
            credentials = self._credentials.model_dump()  # type: ignore
        else:
            credentials = deepcopy(self._credentials)
        for k, v in credentials.items():
            if isinstance(v, Secret):
                credentials[k] = v.get()
            elif isinstance(v, SecretStr):
                credentials[k] = v.get_secret_value()
        return _format_token_from_credentials(urlparse(self._url).netloc, credentials)

    def _add_credentials_to_url(self, url: str) -> str:
        components: ParseResult = urlparse(url)
        credentials = self._formatted_credentials
        if components.scheme != 'https' or not credentials:
            return url
        return urlunparse(components._replace(netloc=f'{credentials}@{components.netloc}'))

    @property
    def _repository_url_with_credentials(self) -> str:
        return self._add_credentials_to_url(self._url)

    @property
    def _git_config(self) -> List[str]:
        config: Dict[str, str] = {}
        if self._include_submodules and self._formatted_credentials:
            base = urlparse(self._url)
            base_url = base._replace(path='')
            without_auth = urlunparse(base_url)
            with_auth = self._add_credentials_to_url(without_auth)
            config[f'url.{with_auth}.insteadOf'] = without_auth
        if config:
            config_str = ' '.join(f'{k}={v}' for k, v in config.items())
            return ['-c', config_str]
        return []

    async def is_sparsely_checked_out(self) -> bool:
        try:
            result = await run_process(['git', 'config', '--get', 'core.sparseCheckout'], cwd=self.destination)
            return result.strip().lower() == 'true'
        except Exception:
            return False

    async def pull_code(self) -> None:
        self._logger.debug("Pulling contents from repository '%s' to '%s'...", self._name, self.destination)
        git_dir: Path = self.destination / '.git'
        if git_dir.exists():
            result = await run_process(['git', 'config', '--get', 'remote.origin.url'], cwd=str(self.destination))
            existing_repo_url: Optional[str] = None
            if result.stdout is not None:
                existing_repo_url = _strip_auth_from_url(result.stdout.decode().strip())
            if existing_repo_url != self._url:
                raise ValueError(f'The existing repository at {str(self.destination)} does not match the configured repository {self._url}')
            if self._directories and (not await self.is_sparsely_checked_out()):
                await run_process(['git', 'sparse-checkout', 'set'] + self._directories, cwd=self.destination)
            self._logger.debug('Pulling latest changes from origin/%s', self._branch)
            cmd: List[str] = ['git']
            cmd += self._git_config
            cmd += ['pull', 'origin']
            if self._branch:
                cmd += [self._branch]
            if self._include_submodules:
                cmd += ['--recurse-submodules']
            cmd += ['--depth', '1']
            try:
                await run_process(cmd, cwd=self.destination)
                self._logger.debug('Successfully pulled latest changes')
            except subprocess.CalledProcessError as exc:
                self._logger.error(f'Failed to pull latest changes with exit code {exc}')
                shutil.rmtree(self.destination)
                await self._clone_repo()
        else:
            await self._clone_repo()

    async def _clone_repo(self) -> None:
        self._logger.debug('Cloning repository %s', self._url)
        repository_url: str = self._repository_url_with_credentials
        cmd: List[str] = ['git']
        cmd += self._git_config
        cmd += ['clone', repository_url]
        if self._branch:
            cmd += ['--branch', self._branch]
        if self._include_submodules:
            cmd += ['--recurse-submodules']
        if self._directories:
            cmd += ['--sparse']
        cmd += ['--depth', '1', str(self.destination)]
        try:
            await run_process(cmd)
        except subprocess.CalledProcessError as exc:
            exc_chain: Optional[Exception] = None if self._credentials else exc
            raise RuntimeError(f'Failed to clone repository {self._url!r} with exit code {exc.returncode}.') from exc_chain
        if self._directories:
            self._logger.debug('Will add %s', self._directories)
            await run_process(['git', 'sparse-checkout', 'set'] + self._directories, cwd=self.destination)

    def __eq__(self, __value: Any) -> bool:
        if isinstance(__value, GitRepository):
            return self._url == __value._url and self._branch == __value._branch and (self._name == __value._name)
        return False

    def __repr__(self) -> str:
        return f'GitRepository(name={self._name!r} repository={self._url!r}, branch={self._branch!r})'

    def to_pull_step(self) -> Dict[str, Any]:
        pull_step: Dict[str, Any] = {'prefect.deployments.steps.git_clone': {'repository': self._url, 'branch': self._branch}}
        if self._include_submodules:
            pull_step['prefect.deployments.steps.git_clone']['include_submodules'] = self._include_submodules
        if isinstance(self._credentials, Block):
            pull_step['prefect.deployments.steps.git_clone']['credentials'] = f'{{{{ {self._credentials.get_block_placeholder()} }}}}'
        elif isinstance(self._credentials, dict):
            token = self._credentials.get('access_token')
            if isinstance(token, Secret):
                pull_step['prefect.deployments.steps.git_clone']['credentials'] = {**self._credentials, 'access_token': f'{{{{ {token.get_block_placeholder()} }}}}'}
            elif self._credentials.get('access_token') is not None:
                raise ValueError('Please save your access token as a Secret block before converting this storage object to a pull step.')
        return pull_step

class RemoteStorage:
    def __init__(self, url: str, pull_interval: Optional[int] = 60, **settings: Any) -> None:
        self._url: str = url
        self._settings: Dict[str, Any] = settings
        self._logger = get_logger('runner.storage.remote-storage')
        self._storage_base_path: Path = Path.cwd()
        self._pull_interval: Optional[int] = pull_interval

    @staticmethod
    def _get_required_package_for_scheme(scheme: str) -> Optional[str]:
        known_implementation = fsspec.registry.get(scheme)
        if known_implementation:
            return known_implementation.__module__.split('.')[0]
        elif scheme == 's3':
            return 's3fs'
        elif scheme in ('gs', 'gcs'):
            return 'gcsfs'
        elif scheme in ('abfs', 'az'):
            return 'adlfs'
        else:
            return None

    @property
    def _filesystem(self) -> Any:
        scheme, _, _, _, _ = urlsplit(self._url)

        def replace_blocks_with_values(obj: Any) -> Any:
            if isinstance(obj, Block):
                if hasattr(obj, 'get'):
                    return obj.get()
                if hasattr(obj, 'value'):
                    return obj.value
                else:
                    return obj.model_dump()
            return obj

        settings_with_block_values = visit_collection(self._settings, replace_blocks_with_values, return_data=True)
        return fsspec.filesystem(scheme, **settings_with_block_values)

    def set_base_path(self, path: Union[str, Path]) -> None:
        self._storage_base_path = Path(path)

    @property
    def pull_interval(self) -> Optional[int]:
        return self._pull_interval

    @property
    def destination(self) -> Path:
        return self._storage_base_path / self._remote_path

    @property
    def _remote_path(self) -> Path:
        _, netloc, urlpath, _, _ = urlsplit(self._url)
        return Path(netloc) / Path(urlpath.lstrip('/'))

    async def pull_code(self) -> None:
        self._logger.debug("Pulling contents from remote storage '%s' to '%s'...", self._url, self.destination)
        if not self.destination.exists():
            self.destination.mkdir(parents=True, exist_ok=True)
        remote_path: str = str(self._remote_path) + '/'
        try:
            await from_async.wait_for_call_in_new_thread(
                create_call(self._filesystem.get, remote_path, str(self.destination), recursive=True)
            )
        except Exception as exc:
            raise RuntimeError(f'Failed to pull contents from remote storage {self._url!r} to {self.destination!r}') from exc

    def to_pull_step(self) -> Dict[str, Any]:
        def replace_block_with_placeholder(obj: Any) -> Any:
            if isinstance(obj, Block):
                return f'{{{{ {obj.get_block_placeholder()} }}}}'
            return obj
        settings_with_placeholders = visit_collection(self._settings, replace_block_with_placeholder, return_data=True)
        required_package: Optional[str] = self._get_required_package_for_scheme(urlparse(self._url).scheme)
        step: Dict[str, Any] = {'prefect.deployments.steps.pull_from_remote_storage': {'url': self._url, **settings_with_placeholders}}
        if required_package:
            step['prefect.deployments.steps.pull_from_remote_storage']['requires'] = required_package
        return step

    def __eq__(self, __value: Any) -> bool:
        if isinstance(__value, RemoteStorage):
            return self._url == __value._url and self._settings == __value._settings
        return False

    def __repr__(self) -> str:
        return f'RemoteStorage(url={self._url!r})'

class BlockStorageAdapter:
    def __init__(self, block: Block, pull_interval: Optional[int] = 60) -> None:
        if not isinstance(block, Block):
            raise TypeError(f'Expected a block object. Received a {type(block).__name__!r} object.')
        if not hasattr(block, 'get_directory'):
            raise ValueError('Provided block must have a `get_directory` method.')
        self._block: Block = block
        self._pull_interval: Optional[int] = pull_interval
        self._storage_base_path: Path = Path.cwd()
        self._name: str = f'{block.get_block_type_slug()}-{block._block_document_name}' if block._block_document_name else str(uuid4())

    def set_base_path(self, path: Union[str, Path]) -> None:
        self._storage_base_path = Path(path)

    @property
    def pull_interval(self) -> Optional[int]:
        return self._pull_interval

    @property
    def destination(self) -> Path:
        return self._storage_base_path / self._name

    async def pull_code(self) -> None:
        if not self.destination.exists():
            self.destination.mkdir(parents=True, exist_ok=True)
        await self._block.get_directory(local_path=str(self.destination))

    def to_pull_step(self) -> Dict[str, Any]:
        if hasattr(self._block, 'get_pull_step'):
            return self._block.get_pull_step()
        else:
            if not self._block._block_document_name:
                raise BlockNotSavedError('Block must be saved with `.save()` before it can be converted to a pull step.')
            return {'prefect.deployments.steps.pull_with_block': {'block_type_slug': self._block.get_block_type_slug(), 'block_document_name': self._block._block_document_name}}

    def __eq__(self, __value: Any) -> bool:
        if isinstance(__value, BlockStorageAdapter):
            return self._block == __value._block
        return False

class LocalStorage:
    def __init__(self, path: Union[str, Path], pull_interval: Optional[int] = None) -> None:
        self._path: Path = Path(path).resolve()
        self._logger = get_logger('runner.storage.local-storage')
        self._storage_base_path: Path = Path.cwd()
        self._pull_interval: Optional[int] = pull_interval

    @property
    def destination(self) -> Path:
        return self._path

    def set_base_path(self, path: Union[str, Path]) -> None:
        self._storage_base_path = Path(path)

    @property
    def pull_interval(self) -> Optional[int]:
        return self._pull_interval

    async def pull_code(self) -> None:
        pass

    def to_pull_step(self) -> Dict[str, Any]:
        step: Dict[str, Any] = {'prefect.deployments.steps.set_working_directory': {'directory': str(self.destination)}}
        return step

    def __eq__(self, __value: Any) -> bool:
        if isinstance(__value, LocalStorage):
            return self._path == __value._path
        return False

    def __repr__(self) -> str:
        return f'LocalStorage(path={self._path!r})'

def create_storage_from_source(source: str, pull_interval: Optional[int] = 60) -> RunnerStorage:
    logger = get_logger('runner.storage')
    parsed_source = urlparse(source)
    if parsed_source.scheme == 'git' or parsed_source.path.endswith('.git'):
        return GitRepository(url=source, pull_interval=pull_interval)
    elif parsed_source.scheme in ('file', 'local'):
        source_path = source.split('://', 1)[-1]
        return LocalStorage(path=source_path, pull_interval=pull_interval)
    elif parsed_source.scheme in fsspec.available_protocols():
        return RemoteStorage(url=source, pull_interval=pull_interval)
    else:
        logger.debug('No valid fsspec protocol found for URL, assuming local storage.')
        return LocalStorage(path=source, pull_interval=pull_interval)

def _format_token_from_credentials(netloc: str, credentials: Optional[Dict[str, Any]]) -> str:
    username: Optional[str] = credentials.get('username') if credentials else None
    password: Optional[Any] = credentials.get('password') if credentials else None
    token: Optional[Any] = credentials.get('token') if credentials else None
    access_token: Optional[Any] = credentials.get('access_token') if credentials else None
    user_provided_token: Optional[Any] = access_token or token or password
    if not user_provided_token:
        raise ValueError('Please provide a `token` or `password` in your Credentials block to clone a repo.')
    if username:
        return f'{username}:{user_provided_token}'
    if 'bitbucketserver' in netloc:
        if not username and ':' not in user_provided_token:
            raise ValueError('Please provide a `username` and a `password` or `token` in your BitBucketCredentials block to clone a repo from BitBucket Server.')
        return f'{username}:{user_provided_token}' if username and username not in user_provided_token else user_provided_token
    elif 'bitbucket' in netloc:
        return user_provided_token if user_provided_token.startswith('x-token-auth:') or ':' in user_provided_token else f'x-token-auth:{user_provided_token}'
    elif 'gitlab' in netloc:
        return f'oauth2:{user_provided_token}' if not user_provided_token.startswith('oauth2:') else user_provided_token
    return user_provided_token

def _strip_auth_from_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc += f':{parsed.port}'
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))