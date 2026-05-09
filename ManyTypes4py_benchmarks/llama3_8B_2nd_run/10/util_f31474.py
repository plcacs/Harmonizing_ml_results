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
from typing import IO, Any, Self, cast

class DecryptError(HomeAssistantError):
    """Error during decryption."""
    _message: str = 'Unexpected error during decryption.'

class EncryptError(HomeAssistantError):
    """Error during encryption."""
    _message: str = 'Unexpected error during encryption.'

class UnsupportedSecureTarVersion(DecryptError):
    """Unsupported securetar version."""
    _message: str = 'Unsupported securetar version.'

class IncorrectPassword(DecryptError):
    """Invalid password or corrupted backup."""
    _message: str = 'Invalid password or corrupted backup.'

class BackupEmpty(DecryptError):
    """No tar files found in the backup."""
    _message: str = 'No tar files found in the backup.'

class AbortCipher(HomeAssistantError):
    """Abort the cipher operation."""
    _message: str = 'Abort cipher operation.'

def make_backup_dir(path: Path) -> None:
    """Create a backup directory if it does not exist."""
    path.mkdir(exist_ok=True)

def read_backup(backup_path: Path) -> AgentBackup:
    """Read a backup from disk."""
    with tarfile.open(backup_path, 'r:', bufsize=BUF_SIZE) as backup_file:
        if not (data_file := backup_file.extractfile('./backup.json')):
            raise KeyError('backup.json not found in tar file')
        data = json_loads_object(data_file.read())
        addons = [AddonInfo(name=cast(str, addon['name']), slug=cast(str, addon['slug']), version=cast(str, addon['version'])) for addon in cast(list[JsonObjectType], data.get('addons', []))]
        folders = [Folder(folder) for folder in cast(list[str], data.get('folders', [])) if folder != 'homeassistant']
        homeassistant_included = False
        homeassistant_version = None
        database_included = False
        if (homeassistant := cast(JsonObjectType, data.get('homeassistant'))) and 'version' in homeassistant:
            homeassistant_included = True
            homeassistant_version = cast(str, homeassistant['version'])
            database_included = not cast(bool, homeassistant.get('exclude_database', False))
        return AgentBackup(addons=addons, backup_id=cast(str, data['slug']), database_included=database_included, date=cast(str, data['date']), extra_metadata=cast(dict[str, bool | str], data.get('extra', {})), folders=folders, homeassistant_included=homeassistant_included, homeassistant_version=homeassistant_version, name=cast(str, data['name']), protected=cast(bool, data.get('protected', False)), size=backup_path.stat().st_size)

def suggested_filename_from_name_date(name: str, date_str: str) -> str:
    """Suggest a filename for the backup."""
    date = dt_util.parse_datetime(date_str, raise_on_error=True)
    return '_'.join(f'{name} {date.strftime("%Y-%m-%d %H.%M %S%f")}.tar'.split())

def suggested_filename(backup: AgentBackup) -> str:
    """Suggest a filename for the backup."""
    return suggested_filename_from_name_date(backup.name, backup.date)

def validate_password(path: Path, password: str | None) -> bool:
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
            with SecureTarFile(path, gzip=compressed, key=password_to_key(password) if password is not None else None, mode='r', fileobj=ha_tar):
                return True
        except tarfile.ReadError:
            LOGGER.debug('Invalid password')
            return False
        except Exception:
            LOGGER.exception('Unexpected error validating password')
    return False

class AsyncIteratorReader:
    """Wrap an AsyncIterator."""

    def __init__(self, hass: HomeAssistant, stream: AsyncIterator) -> None:
        """Initialize the wrapper."""
        self._aborted = False
        self._hass = hass
        self._stream = stream
        self._buffer = None
        self._next_future = None
        self._pos = 0

    async def _next(self) -> Any:
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
        while n < 0 or len(result) < n:
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
            chunk = self._buffer[self._pos:self._pos + n]
            result.extend(chunk)
            n -= len(chunk)
            self._pos += len(chunk)
            if self._pos == len(self._buffer):
                self._buffer = None
        return bytes(result)

    def close(self) -> None:
        """Close the iterator."""

class AsyncIteratorWriter:
    """Wrap an AsyncIterator."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the wrapper."""
        self._aborted = False
        self._hass = hass
        self._pos = 0
        self._queue = asyncio.Queue(maxsize=1)
        self._write_future = None

    def __aiter__(self) -> Self:
        """Return the iterator."""
        return self

    async def __anext__(self) -> Any:
        """Get the next chunk from the iterator."""
        if (data := (await self._queue.get())):
            return data
        raise StopAsyncIteration

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

def validate_password_stream(input_stream: IO, password: str | None) -> None:
    """Decrypt a backup."""
    with tarfile.open(fileobj=input_stream, mode='r|', bufsize=BUF_SIZE) as input_tar:
        for obj in input_tar:
            if not obj.name.endswith(('.tar', '.tgz', '.tar.gz')):
                continue
            istf = SecureTarFile(None, gzip=False, key=password_to_key(password) if password is not None else None, mode='r', fileobj=input_tar.extractfile(obj))
            with istf.decrypt(obj) as decrypted:
                if istf.securetar_header.plaintext_size is None:
                    raise UnsupportedSecureTarVersion
                try:
                    decrypted.read(1)
                except SecureTarReadError as err:
                    raise IncorrectPassword from err
                return
    raise BackupEmpty

def decrypt_backup(input_stream: IO, output_stream: IO, password: str | None, on_done: Callable[[Exception | None], None], minimum_size: int, nonces: list[bytes]) -> None:
    """Decrypt a backup."""
    error = None
    try:
        try:
            with tarfile.open(fileobj=input_stream, mode='r|', bufsize=BUF_SIZE) as input_tar, tarfile.open(fileobj=output_stream, mode='w|', bufsize=BUF_SIZE) as output_tar:
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

def _decrypt_backup(input_tar: tarfile.TarFile, output_tar: tarfile.TarFile, password: str | None) -> None:
    """Decrypt a backup."""
    for obj in input_tar:
        if PurePath(obj.name) == PurePath('backup.json'):
            if not (reader := input_tar.extractfile(obj)):
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
        istf = SecureTarFile(None, gzip=False, key=password_to_key(password) if password is not None else None, mode='r', fileobj=input_tar.extractfile(obj))
        with istf.decrypt(obj) as decrypted:
            if (plaintext_size := istf.securetar_header.plaintext_size) is None:
                raise UnsupportedSecureTarVersion
            decrypted_obj = copy.deepcopy(obj)
            decrypted_obj.size = plaintext_size
            output_tar.addfile(decrypted_obj, decrypted)

def encrypt_backup(input_stream: IO, output_stream: IO, password: str, on_done: Callable[[Exception | None], None], minimum_size: int, nonces: list[bytes]) -> None:
    """Encrypt a backup."""
    error = None
    try:
        try:
            with tarfile.open(fileobj=input_stream, mode='r|', bufsize=BUF_SIZE) as input_tar, tarfile.open(fileobj=output_stream, mode='w|', bufsize=BUF_SIZE) as output_tar:
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

def _encrypt_backup(input_tar: tarfile.TarFile, output_tar: tarfile.TarFile, password: str, nonces: list[bytes]) -> None:
    """Encrypt a backup."""
    inner_tar_idx = 0
    for obj in input_tar:
        if PurePath(obj.name) == PurePath('backup.json'):
            if not (reader := input_tar.extractfile(obj)):
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
        istf = SecureTarFile(None, gzip=False, key=password_to_key(password) if password is not None else None, mode='r', fileobj=input_tar.extractfile(obj), nonce=nonces[inner_tar_idx])
        inner_tar_idx += 1
        with istf.encrypt(obj) as encrypted:
            encrypted_obj = copy.deepcopy(obj)
            encrypted_obj.size = encrypted.encrypted_size
            output_tar.addfile(encrypted_obj, encrypted)

async def receive_file(hass: HomeAssistant, contents: AsyncIterator, path: Path) -> None:
    """Receive a file from a stream and write it to a file."""
    queue = SimpleQueue()

    def _sync_queue_consumer() -> None:
        with path.open('wb') as file_handle:
            while True:
                if (_chunk_future := queue.get()) is None:
                    break
                _chunk, _future = _chunk_future
                if _future is not None:
                    hass.loop.call_soon_threadsafe(_future.set_result, None)
                file_handle.write(_chunk)

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

class DecryptedBackupStreamer(_CipherBackupStreamer):
    """Decrypt a backup."""
    _cipher_func = staticmethod(decrypt_backup)

    def backup(self) -> AgentBackup:
        """Return the decrypted backup."""
        return replace(self._backup, protected=False, size=self.size())

class EncryptedBackupStreamer(_CipherBackupStreamer):
    """Encrypt a backup."""

    def __init__(self, hass: HomeAssistant, backup: AgentBackup, open_stream: Callable[[Path], IO], password: str) -> None:
        """Initialize."""
        super().__init__(hass, backup, open_stream, password)
        self._nonces = [os.urandom(16) for _ in range(self._num_tar_files())]
    _cipher_func = staticmethod(encrypt_backup)

    def backup(self) -> AgentBackup:
        """Return the encrypted backup."""
        return replace(self._backup, protected=True, size=self.size())

async def main(hass: HomeAssistant) -> None:
    """Main function."""
    # Initialize
    backup = AgentBackup(addons=[], backup_id='backup', database_included=True, date='2022-01-01 12:00:00', extra_metadata={}, folders=['homeassistant'], homeassistant_included=True, homeassistant_version='12.0', name='homeassistant', protected=False, size=1024)
    open_stream = lambda path: open(path, 'wb')
    password = 'password123'
    streamer = DecryptedBackupStreamer(hass, backup, open_stream, password)
    # Use the streamer
    stream = await streamer.open_stream()
    await receive_file(hass, stream, Path('path/to/file'))
    # Wait for the worker threads to finish
    await streamer.wait()
