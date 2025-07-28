from __future__ import annotations
import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from concurrent.futures import CancelledError, Future
import copy
from dataclasses import dataclass, replace
from io import BytesIO
import json
import os
from pathlib import Path, PurePath
from queue import SimpleQueue
import tarfile
import threading
from typing import IO, Any, Optional, cast, List, Awaitable
import aiohttp
from securetar import SecureTarError, SecureTarFile, SecureTarReadError
from homeassistant.backup_restore import password_to_key
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType, json_loads_object
from .const import BUF_SIZE, LOGGER
from .models import AddonInfo, AgentBackup, Folder

class DecryptError(HomeAssistantError):
    """Error during decryption."""
    _message = 'Unexpected error during decryption.'

class EncryptError(HomeAssistantError):
    """Error during encryption."""
    _message = 'Unexpected error during encryption.'

class UnsupportedSecureTarVersion(DecryptError):
    """Unsupported securetar version."""
    _message = 'Unsupported securetar version.'

class IncorrectPassword(DecryptError):
    """Invalid password or corrupted backup."""
    _message = 'Invalid password or corrupted backup.'

class BackupEmpty(DecryptError):
    """No tar files found in the backup."""
    _message = 'No tar files found in the backup.'

class AbortCipher(HomeAssistantError):
    """Abort the cipher operation."""
    _message = 'Abort cipher operation.'

def make_backup_dir(path: Path) -> None:
    """Create a backup directory if it does not exist."""
    path.mkdir(exist_ok=True)

def read_backup(backup_path: Path) -> AgentBackup:
    """Read a backup from disk."""
    with tarfile.open(backup_path, 'r:', bufsize=BUF_SIZE) as backup_file:
        if not (data_file := backup_file.extractfile('./backup.json')):
            raise KeyError('backup.json not found in tar file')
        data = json_loads_object(data_file.read())
        addons: List[AddonInfo] = [
            AddonInfo(name=cast(str, addon['name']),
                      slug=cast(str, addon['slug']),
                      version=cast(str, addon['version']))
            for addon in cast(list[JsonObjectType], data.get('addons', []))
        ]
        folders: List[Folder] = [Folder(folder) for folder in cast(list[str], data.get('folders', [])) if folder != 'homeassistant']
        homeassistant_included = False
        homeassistant_version: Optional[str] = None
        database_included = False
        if (homeassistant := cast(JsonObjectType, data.get('homeassistant'))) and 'version' in homeassistant:
            homeassistant_included = True
            homeassistant_version = cast(str, homeassistant['version'])
            database_included = not cast(bool, homeassistant.get('exclude_database', False))
        return AgentBackup(
            addons=addons,
            backup_id=cast(str, data['slug']),
            database_included=database_included,
            date=cast(str, data['date']),
            extra_metadata=cast(dict[str, bool | str], data.get('extra', {})),
            folders=folders,
            homeassistant_included=homeassistant_included,
            homeassistant_version=homeassistant_version,
            name=cast(str, data['name']),
            protected=cast(bool, data.get('protected', False)),
            size=backup_path.stat().st_size
        )

def suggested_filename_from_name_date(name: str, date_str: str) -> str:
    """Suggest a filename for the backup."""
    date = dt_util.parse_datetime(date_str, raise_on_error=True)
    return '_'.join(f'{name} {date.strftime("%Y-%m-%d %H.%M %S%f")}.tar'.split())

def suggested_filename(backup: AgentBackup) -> str:
    """Suggest a filename for the backup."""
    return suggested_filename_from_name_date(backup.name, backup.date)

def validate_password(path: Path, password: Optional[str]) -> bool:
    """Validate the password."""
    with tarfile.open(path, 'r:', bufsize=BUF_SIZE) as backup_file:
        compressed = False
        ha_tar_name = 'homeassistant.tar'
        try:
            ha_tar = backup_file.extractfile(ha_tar_name)
        except KeyError:
            compressed = True
            ha_tar_name = 'homeassistant.tar.gz'
            try:
                ha_tar = backup_file.extractfile(ha_tar_name)
            except KeyError:
                LOGGER.error('No homeassistant.tar or homeassistant.tar.gz found')
                return False
        try:
            with SecureTarFile(
                path,
                gzip=compressed,
                key=password_to_key(password) if password is not None else None,
                mode='r',
                fileobj=ha_tar
            ):
                return True
        except tarfile.ReadError:
            LOGGER.debug('Invalid password')
            return False
        except Exception:
            LOGGER.exception('Unexpected error validating password')
    return False

class AsyncIteratorReader:
    """Wrap an AsyncIterator."""
    _aborted: bool
    _hass: HomeAssistant
    _stream: AsyncIterator[bytes]
    _buffer: Optional[bytes]
    _next_future: Optional[Future]
    _pos: int

    def __init__(self, hass: HomeAssistant, stream: AsyncIterator[bytes]) -> None:
        """Initialize the wrapper."""
        self._aborted = False
        self._hass = hass
        self._stream = stream
        self._buffer = None
        self._next_future = None
        self._pos = 0

    async def _next(self) -> Optional[bytes]:
        """Get the next chunk from the iterator."""
        return await anext(self._stream, None)

    def abort(self) -> None:
        """Abort the reader."""
        self._aborted = True
        if self._next_future is not None:
            self._next_future.cancel()

    def read(self, n: int = -1) -> bytes:
        """Read data from the iterator."""
        result = bytearray()
        remaining = n
        while remaining < 0 or len(result) < n:
            if not self._buffer:
                self._next_future = asyncio.run_coroutine_threadsafe(self._next(), self._hass.loop)
                if self._aborted:
                    self._next_future.cancel()
                    raise AbortCipher
                try:
                    self._buffer = self._next_future.result()
                except CancelledError as err:
                    raise AbortCipher from err
                self._pos = 0
            if not self._buffer:
                break
            if remaining < 0:
                chunk = self._buffer[self._pos :]
            else:
                chunk = self._buffer[self._pos : self._pos + remaining]
            result.extend(chunk)
            if remaining > 0:
                remaining -= len(chunk)
            self._pos += len(chunk)
            if self._pos == len(self._buffer):
                self._buffer = None
        return bytes(result)

    def close(self) -> None:
        """Close the iterator."""
        pass

class AsyncIteratorWriter:
    """Wrap an AsyncIterator."""
    _aborted: bool
    _hass: HomeAssistant
    _pos: int
    _queue: asyncio.Queue[Optional[bytes]]
    _write_future: Optional[Future]

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the wrapper."""
        self._aborted = False
        self._hass = hass
        self._pos = 0
        self._queue = asyncio.Queue(maxsize=1)
        self._write_future = None

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Return the iterator."""
        return self

    async def __anext__(self) -> bytes:
        """Get the next chunk from the iterator."""
        data = await self._queue.get()
        if data is None:
            raise StopAsyncIteration
        return data

    def abort(self) -> None:
        """Abort the writer."""
        self._aborted = True
        if self._write_future is not None:
            self._write_future.cancel()

    def tell(self) -> int:
        """Return the current position in the iterator."""
        return self._pos

    def write(self, s: bytes) -> int:
        """Write data to the iterator."""
        self._write_future = asyncio.run_coroutine_threadsafe(self._queue.put(s), self._hass.loop)
        if self._aborted:
            self._write_future.cancel()
            raise AbortCipher
        try:
            self._write_future.result()
        except CancelledError as err:
            raise AbortCipher from err
        self._pos += len(s)
        return len(s)

def validate_password_stream(input_stream: IO[bytes], password: Optional[str]) -> None:
    """Decrypt a backup."""
    with tarfile.open(fileobj=input_stream, mode='r|', bufsize=BUF_SIZE) as input_tar:
        for obj in input_tar:
            if not obj.name.endswith(('.tar', '.tgz', '.tar.gz')):
                continue
            istf = SecureTarFile(
                None,
                gzip=False,
                key=password_to_key(password) if password is not None else None,
                mode='r',
                fileobj=input_tar.extractfile(obj)
            )
            with istf.decrypt(obj) as decrypted:
                if istf.securetar_header.plaintext_size is None:
                    raise UnsupportedSecureTarVersion
                try:
                    decrypted.read(1)
                except SecureTarReadError as err:
                    raise IncorrectPassword from err
                return
    raise BackupEmpty

def decrypt_backup(
    input_stream: IO[bytes],
    output_stream: IO[bytes],
    password: Optional[str],
    on_done: Callable[[Optional[Exception]], None],
    minimum_size: int,
    nonces: List[bytes]
) -> None:
    """Decrypt a backup."""
    error: Optional[Exception] = None
    try:
        try:
            with tarfile.open(fileobj=input_stream, mode='r|', bufsize=BUF_SIZE) as input_tar, \
                 tarfile.open(fileobj=output_stream, mode='w|', bufsize=BUF_SIZE) as output_tar:
                _decrypt_backup(input_tar, output_tar, password)
        except (DecryptError, SecureTarError, tarfile.TarError) as err:
            LOGGER.warning('Error decrypting backup: %s', err)
            error = err
        else:
            padding = max(minimum_size - output_stream.tell(), 0)
            output_stream.write(b'\x00' * padding)
        finally:
            output_stream.write(b'')
    except AbortCipher:
        LOGGER.debug('Cipher operation aborted')
    finally:
        on_done(error)

def _decrypt_backup(
    input_tar: tarfile.TarFile,
    output_tar: tarfile.TarFile,
    password: Optional[str]
) -> None:
    """Decrypt a backup."""
    for obj in input_tar:
        if PurePath(obj.name) == PurePath('backup.json'):
            reader = input_tar.extractfile(obj)
            if not reader:
                raise DecryptError
            metadata = json_loads_object(reader.read())
            metadata['protected'] = False
            updated_metadata_b = json.dumps(metadata).encode()
            metadata_obj = copy.deepcopy(obj)
            metadata_obj.size = len(updated_metadata_b)
            output_tar.addfile(metadata_obj, BytesIO(updated_metadata_b))
            continue
        if not obj.name.endswith(('.tar', '.tgz', '.tar.gz')):
            output_tar.addfile(obj, input_tar.extractfile(obj))
            continue
        istf = SecureTarFile(
            None,
            gzip=False,
            key=password_to_key(password) if password is not None else None,
            mode='r',
            fileobj=input_tar.extractfile(obj)
        )
        with istf.decrypt(obj) as decrypted:
            plaintext_size = istf.securetar_header.plaintext_size
            if plaintext_size is None:
                raise UnsupportedSecureTarVersion
            decrypted_obj = copy.deepcopy(obj)
            decrypted_obj.size = plaintext_size
            output_tar.addfile(decrypted_obj, decrypted)

def encrypt_backup(
    input_stream: IO[bytes],
    output_stream: IO[bytes],
    password: Optional[str],
    on_done: Callable[[Optional[Exception]], None],
    minimum_size: int,
    nonces: List[bytes]
) -> None:
    """Encrypt a backup."""
    error: Optional[Exception] = None
    try:
        try:
            with tarfile.open(fileobj=input_stream, mode='r|', bufsize=BUF_SIZE) as input_tar, \
                 tarfile.open(fileobj=output_stream, mode='w|', bufsize=BUF_SIZE) as output_tar:
                _encrypt_backup(input_tar, output_tar, password, nonces)
        except (EncryptError, SecureTarError, tarfile.TarError) as err:
            LOGGER.warning('Error encrypting backup: %s', err)
            error = err
        else:
            padding = max(minimum_size - output_stream.tell(), 0)
            output_stream.write(b'\x00' * padding)
        finally:
            output_stream.write(b'')
    except AbortCipher:
        LOGGER.debug('Cipher operation aborted')
    finally:
        on_done(error)

def _encrypt_backup(
    input_tar: tarfile.TarFile,
    output_tar: tarfile.TarFile,
    password: Optional[str],
    nonces: List[bytes]
) -> None:
    """Encrypt a backup."""
    inner_tar_idx = 0
    for obj in input_tar:
        if PurePath(obj.name) == PurePath('backup.json'):
            reader = input_tar.extractfile(obj)
            if not reader:
                raise EncryptError
            metadata = json_loads_object(reader.read())
            metadata['protected'] = True
            updated_metadata_b = json.dumps(metadata).encode()
            metadata_obj = copy.deepcopy(obj)
            metadata_obj.size = len(updated_metadata_b)
            output_tar.addfile(metadata_obj, BytesIO(updated_metadata_b))
            continue
        if not obj.name.endswith(('.tar', '.tgz', '.tar.gz')):
            output_tar.addfile(obj, input_tar.extractfile(obj))
            continue
        istf = SecureTarFile(
            None,
            gzip=False,
            key=password_to_key(password) if password is not None else None,
            mode='r',
            fileobj=input_tar.extractfile(obj),
            nonce=nonces[inner_tar_idx]
        )
        inner_tar_idx += 1
        with istf.encrypt(obj) as encrypted:
            encrypted_obj = copy.deepcopy(obj)
            encrypted_obj.size = encrypted.encrypted_size
            output_tar.addfile(encrypted_obj, encrypted)

@dataclass(kw_only=True)
class _CipherWorkerStatus:
    error: Optional[Exception] = None
    done: asyncio.Event[Any] = asyncio.Event()
    reader: AsyncIteratorReader = None  # type: ignore
    thread: threading.Thread = None  # type: ignore
    writer: AsyncIteratorWriter = None  # type: ignore

class _CipherBackupStreamer:
    """Encrypt or decrypt a backup."""
    _hass: HomeAssistant
    _backup: AgentBackup
    _open_stream: Callable[[], Coroutine[Any, Any, AsyncIterator[bytes]]]
    _password: Optional[str]
    _nonces: List[bytes]
    _workers: List[_CipherWorkerStatus]

    def __init__(
        self,
        hass: HomeAssistant,
        backup: AgentBackup,
        open_stream: Callable[[], Coroutine[Any, Any, AsyncIterator[bytes]]],
        password: Optional[str]
    ) -> None:
        """Initialize."""
        self._workers = []
        self._backup = backup
        self._hass = hass
        self._open_stream = open_stream
        self._password = password
        self._nonces = []

    def size(self) -> int:
        """Return the maximum size of the decrypted or encrypted backup."""
        return self._backup.size + self._num_tar_files() * tarfile.RECORDSIZE

    def _num_tar_files(self) -> int:
        """Return the number of inner tar files."""
        b = self._backup
        return len(b.addons) + len(b.folders) + (1 if b.homeassistant_included else 0) + 1

    async def open_stream(self) -> AsyncIteratorWriter:
        """Open a stream."""
        stream: AsyncIterator[bytes] = await self._open_stream()

        reader = AsyncIteratorReader(self._hass, stream)
        writer = AsyncIteratorWriter(self._hass)

        def on_done(error: Optional[Exception]) -> None:
            """Called by the worker thread when it is done."""
            worker_status.error = error
            self._hass.loop.call_soon_threadsafe(worker_status.done.set)

        worker = threading.Thread(target=self._cipher_func, args=[reader, writer, self._password, on_done, self.size(), self._nonces])
        worker_status = _CipherWorkerStatus(done=asyncio.Event(), reader=reader, thread=worker, writer=writer)
        self._workers.append(worker_status)
        worker.start()
        return writer

    async def wait(self) -> None:
        """Wait for the worker threads to finish."""
        for worker in self._workers:
            worker.reader.abort()
            worker.writer.abort()
        await asyncio.gather(*(worker.done.wait() for worker in self._workers))

class DecryptedBackupStreamer(_CipherBackupStreamer):
    """Decrypt a backup."""
    _cipher_func = staticmethod(decrypt_backup)

    def backup(self) -> AgentBackup:
        """Return the decrypted backup."""
        return replace(self._backup, protected=False, size=self.size())

class EncryptedBackupStreamer(_CipherBackupStreamer):
    """Encrypt a backup."""
    _cipher_func: Callable = None

    def __init__(
        self,
        hass: HomeAssistant,
        backup: AgentBackup,
        open_stream: Callable[[], Coroutine[Any, Any, AsyncIterator[bytes]]],
        password: Optional[str]
    ) -> None:
        """Initialize."""
        super().__init__(hass, backup, open_stream, password)
        self._nonces = [os.urandom(16) for _ in range(self._num_tar_files())]

    _cipher_func = staticmethod(encrypt_backup)

    def backup(self) -> AgentBackup:
        """Return the encrypted backup."""
        return replace(self._backup, protected=True, size=self.size())

async def receive_file(hass: HomeAssistant, contents: Any, path: Path) -> None:
    """Receive a file from a stream and write it to a file."""
    queue: SimpleQueue = SimpleQueue()

    def _sync_queue_consumer() -> None:
        with path.open('wb') as file_handle:
            while True:
                item = queue.get()
                if item is None:
                    break
                chunk, future = item  # type: (bytes, Optional[asyncio.Future])
                if future is not None:
                    hass.loop.call_soon_threadsafe(future.set_result, None)
                file_handle.write(chunk)

    fut = None
    try:
        fut = hass.async_add_executor_job(_sync_queue_consumer)
        megabytes_sending = 0
        while (chunk := (await contents.read_chunk(BUF_SIZE))):
            megabytes_sending += 1
            if megabytes_sending % 5 != 0:
                queue.put_nowait((chunk, None))
                continue
            chunk_future = hass.loop.create_future()
            queue.put_nowait((chunk, chunk_future))
            await asyncio.wait((fut, chunk_future), return_when=asyncio.FIRST_COMPLETED)
            if fut.done():
                break
        queue.put_nowait(None)
    finally:
        if fut is not None:
            await fut