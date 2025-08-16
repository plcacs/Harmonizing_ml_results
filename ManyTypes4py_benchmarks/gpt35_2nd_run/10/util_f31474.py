from __future__ import annotations
import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import Future
import copy
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path, PurePath
from queue import SimpleQueue
import tarfile
import threading
from typing import IO, Any, cast
import aiohttp
from securetar import SecureTarError, SecureTarFile, SecureTarReadError
from homeassistant.backup_restore import password_to_key
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType
from .const import BUF_SIZE, LOGGER
from .models import AddonInfo, AgentBackup, Folder

@dataclass
class DecryptError(HomeAssistantError):
    _message: str = 'Unexpected error during decryption.'

@dataclass
class EncryptError(HomeAssistantError):
    _message: str = 'Unexpected error during encryption.'

@dataclass
class UnsupportedSecureTarVersion(DecryptError):
    _message: str = 'Unsupported securetar version.'

@dataclass
class IncorrectPassword(DecryptError):
    _message: str = 'Invalid password or corrupted backup.'

@dataclass
class BackupEmpty(DecryptError):
    _message: str = 'No tar files found in the backup.'

@dataclass
class AbortCipher(HomeAssistantError):
    _message: str = 'Abort cipher operation.'

def make_backup_dir(path: Path) -> None:
    path.mkdir(exist_ok=True)

def read_backup(backup_path: Path) -> AgentBackup:
    ...

def suggested_filename_from_name_date(name: str, date_str: str) -> str:
    ...

def suggested_filename(backup: AgentBackup) -> str:
    ...

def validate_password(path: Path, password: str) -> bool:
    ...

class AsyncIteratorReader:
    ...

class AsyncIteratorWriter:
    ...

def validate_password_stream(input_stream: IO[bytes], password: str) -> None:
    ...

def decrypt_backup(input_stream: IO[bytes], output_stream: IO[bytes], password: str, on_done: Callable, minimum_size: int, nonces: list[bytes]) -> None:
    ...

def encrypt_backup(input_stream: IO[bytes], output_stream: IO[bytes], password: str, on_done: Callable, minimum_size: int, nonces: list[bytes]) -> None:
    ...

@dataclass
class _CipherWorkerStatus:
    error: Any = None

class _CipherBackupStreamer:
    ...

class DecryptedBackupStreamer(_CipherBackupStreamer):
    ...

class EncryptedBackupStreamer(_CipherBackupStreamer):
    ...

async def receive_file(hass: HomeAssistant, contents: AsyncIterator[bytes], path: Path) -> None:
    ...
