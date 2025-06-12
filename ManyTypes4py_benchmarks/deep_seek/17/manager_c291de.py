"""Backup manager for the Backup integration."""
from __future__ import annotations
import abc
import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import io
from itertools import chain
import json
from pathlib import Path, PurePath
import shutil
import tarfile
import time
from typing import IO, TYPE_CHECKING, Any, Protocol, TypedDict, cast, Optional, Union, Dict, List, Set, Tuple, Awaitable, TypeVar, Generic, Type, Literal

import aiohttp
from securetar import SecureTarFile, atomic_contents_add
from homeassistant.backup_restore import RESTORE_BACKUP_FILE, RESTORE_BACKUP_RESULT_FILE, password_to_key
from homeassistant.const import __version__ as HAVERSION
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import instance_id, integration_platform, issue_registry as ir
from homeassistant.helpers.json import json_bytes
from homeassistant.util import dt as dt_util, json as json_util

from . import util as backup_util
from .agent import BackupAgent, BackupAgentError, BackupAgentPlatformProtocol, LocalBackupAgent
from .config import BackupConfig, delete_backups_exceeding_configured_count
from .const import BUF_SIZE, DATA_MANAGER, DOMAIN, EXCLUDE_DATABASE_FROM_BACKUP, EXCLUDE_FROM_BACKUP, LOGGER
from .models import AgentBackup, BackupError, BackupManagerError, BackupReaderWriterError, BaseBackup, Folder
from .store import BackupStore
from .util import AsyncIteratorReader, DecryptedBackupStreamer, EncryptedBackupStreamer, make_backup_dir, read_backup, validate_password, validate_password_stream

T = TypeVar('T')
BackupFilter = Callable[[Dict[str, 'ManagerBackup']], Dict[str, 'ManagerBackup']]

@dataclass(frozen=True, kw_only=True, slots=True)
class NewBackup:
    """New backup class."""
    backup_job_id: str

@dataclass(frozen=True, kw_only=True, slots=True)
class AgentBackupStatus:
    """Agent specific backup attributes."""
    protected: bool
    size: int

@dataclass(frozen=True, kw_only=True, slots=True)
class ManagerBackup(BaseBackup):
    """Backup class."""
    agents: Dict[str, AgentBackupStatus]
    failed_agent_ids: List[str]
    with_automatic_settings: Optional[bool]

@dataclass(frozen=True, kw_only=True, slots=True)
class WrittenBackup:
    """Written backup class."""
    backup: AgentBackup
    open_stream: Callable[[], Awaitable[AsyncIterator[bytes]]]
    release_stream: Callable[[], Awaitable[None]]

class BackupManagerState(StrEnum):
    """Backup state type."""
    IDLE = 'idle'
    CREATE_BACKUP = 'create_backup'
    RECEIVE_BACKUP = 'receive_backup'
    RESTORE_BACKUP = 'restore_backup'

class CreateBackupStage(StrEnum):
    """Create backup stage enum."""
    ADDON_REPOSITORIES = 'addon_repositories'
    ADDONS = 'addons'
    AWAIT_ADDON_RESTARTS = 'await_addon_restarts'
    DOCKER_CONFIG = 'docker_config'
    FINISHING_FILE = 'finishing_file'
    FOLDERS = 'folders'
    HOME_ASSISTANT = 'home_assistant'
    UPLOAD_TO_AGENTS = 'upload_to_agents'

class CreateBackupState(StrEnum):
    """Create backup state enum."""
    COMPLETED = 'completed'
    FAILED = 'failed'
    IN_PROGRESS = 'in_progress'

class ReceiveBackupStage(StrEnum):
    """Receive backup stage enum."""
    RECEIVE_FILE = 'receive_file'
    UPLOAD_TO_AGENTS = 'upload_to_agents'

class ReceiveBackupState(StrEnum):
    """Receive backup state enum."""
    COMPLETED = 'completed'
    FAILED = 'failed'
    IN_PROGRESS = 'in_progress'

class RestoreBackupStage(StrEnum):
    """Restore backup stage enum."""
    ADDON_REPOSITORIES = 'addon_repositories'
    ADDONS = 'addons'
    AWAIT_ADDON_RESTARTS = 'await_addon_restarts'
    AWAIT_HOME_ASSISTANT_RESTART = 'await_home_assistant_restart'
    CHECK_HOME_ASSISTANT = 'check_home_assistant'
    DOCKER_CONFIG = 'docker_config'
    DOWNLOAD_FROM_AGENT = 'download_from_agent'
    FOLDERS = 'folders'
    HOME_ASSISTANT = 'home_assistant'
    REMOVE_DELTA_ADDONS = 'remove_delta_addons'

class RestoreBackupState(StrEnum):
    """Receive backup state enum."""
    COMPLETED = 'completed'
    CORE_RESTART = 'core_restart'
    FAILED = 'failed'
    IN_PROGRESS = 'in_progress'

@dataclass(frozen=True, kw_only=True, slots=True)
class ManagerStateEvent:
    """Backup state class."""
    manager_state: BackupManagerState
    stage: Optional[Union[CreateBackupStage, ReceiveBackupStage, RestoreBackupStage]] = None
    state: Optional[Union[CreateBackupState, ReceiveBackupState, RestoreBackupState]] = None
    reason: Optional[str] = None

@dataclass(frozen=True, kw_only=True, slots=True)
class IdleEvent(ManagerStateEvent):
    """Backup manager idle."""
    manager_state: Literal[BackupManagerState.IDLE] = BackupManagerState.IDLE

@dataclass(frozen=True, kw_only=True, slots=True)
class CreateBackupEvent(ManagerStateEvent):
    """Backup in progress."""
    manager_state: Literal[BackupManagerState.CREATE_BACKUP] = BackupManagerState.CREATE_BACKUP

@dataclass(frozen=True, kw_only=True, slots=True)
class ReceiveBackupEvent(ManagerStateEvent):
    """Backup receive."""
    manager_state: Literal[BackupManagerState.RECEIVE_BACKUP] = BackupManagerState.RECEIVE_BACKUP

@dataclass(frozen=True, kw_only=True, slots=True)
class RestoreBackupEvent(ManagerStateEvent):
    """Backup restore."""
    manager_state: Literal[BackupManagerState.RESTORE_BACKUP] = BackupManagerState.RESTORE_BACKUP

class BackupPlatformProtocol(Protocol):
    """Define the format that backup platforms can have."""

    async def async_pre_backup(self, hass: HomeAssistant) -> None:
        """Perform operations before a backup starts."""

    async def async_post_backup(self, hass: HomeAssistant) -> None:
        """Perform operations after a backup finishes."""

class BackupReaderWriter(abc.ABC):
    """Abstract class for reading and writing backups."""

    @abc.abstractmethod
    async def async_create_backup(
        self,
        *,
        agent_ids: List[str],
        backup_name: str,
        extra_metadata: Dict[str, Any],
        include_addons: List[str],
        include_all_addons: bool,
        include_database: bool,
        include_folders: List[str],
        include_homeassistant: bool,
        on_progress: Callable[[ManagerStateEvent], None],
        password: Optional[str]
    ) -> Tuple[NewBackup, asyncio.Task[WrittenBackup]]:
        """Create a backup."""

    @abc.abstractmethod
    async def async_receive_backup(
        self,
        *,
        agent_ids: List[str],
        stream: aiohttp.StreamReader,
        suggested_filename: str
    ) -> WrittenBackup:
        """Receive a backup."""

    @abc.abstractmethod
    async def async_restore_backup(
        self,
        backup_id: str,
        *,
        agent_id: str,
        on_progress: Callable[[ManagerStateEvent], None],
        open_stream: Callable[[], Awaitable[AsyncIterator[bytes]]],
        password: Optional[str],
        restore_addons: List[str],
        restore_database: bool,
        restore_folders: List[str],
        restore_homeassistant: bool
    ) -> None:
        """Restore a backup."""

    @abc.abstractmethod
    async def async_resume_restore_progress_after_restart(
        self,
        *,
        on_progress: Callable[[ManagerStateEvent], None]
    ) -> None:
        """Get restore events after core restart."""

class IncorrectPasswordError(BackupReaderWriterError):
    """Raised when the password is incorrect."""
    error_code: str = 'password_incorrect'
    _message: str = 'The password provided is incorrect.'

class DecryptOnDowloadNotSupported(BackupManagerError):
    """Raised when on-the-fly decryption is not supported."""
    error_code: str = 'decrypt_on_download_not_supported'
    _message: str = 'On-the-fly decryption is not supported for this backup.'

class BackupManager:
    """Define the format that backup managers can have."""

    def __init__(self, hass: HomeAssistant, reader_writer: BackupReaderWriter) -> None:
        """Initialize the backup manager."""
        self.hass: HomeAssistant = hass
        self.platforms: Dict[str, BackupPlatformProtocol] = {}
        self.backup_agent_platforms: Dict[str, BackupAgentPlatformProtocol] = {}
        self.backup_agents: Dict[str, BackupAgent] = {}
        self.local_backup_agents: Dict[str, LocalBackupAgent] = {}
        self.config: BackupConfig = BackupConfig(hass, self)
        self._reader_writer: BackupReaderWriter = reader_writer
        self.known_backups: KnownBackups = KnownBackups(self)
        self.store: BackupStore = BackupStore(hass, self)
        self._backup_task: Optional[asyncio.Task[WrittenBackup]] = None
        self._backup_finish_task: Optional[asyncio.Task[None]] = None
        self.remove_next_backup_event: Optional[Callable[[], None]] = None
        self.remove_next_delete_event: Optional[Callable[[], None]] = None
        self.last_event: ManagerStateEvent = IdleEvent()
        self.last_non_idle_event: Optional[ManagerStateEvent] = None
        self._backup_event_subscriptions: List[Callable[[ManagerStateEvent], None]] = []

    async def async_setup(self) -> None:
        """Set up the backup manager."""
        stored = await self.store.load()
        if stored:
            self.config.load(stored['config'])
            self.known_backups.load(stored['backups'])
        await self._reader_writer.async_resume_restore_progress_after_restart(on_progress=self.async_on_backup_event)
        await self.load_platforms()

    @property
    def state(self) -> BackupManagerState:
        """Return the state of the backup manager."""
        return self.last_event.manager_state

    @callback
    def _add_platform_pre_post_handler(self, integration_domain: str, platform: Any) -> None:
        """Add a backup platform."""
        if not hasattr(platform, 'async_pre_backup') or not hasattr(platform, 'async_post_backup'):
            return
        self.platforms[integration_domain] = platform

    @callback
    def _async_add_backup_agent_platform(self, integration_domain: str, platform: Any) -> None:
        """Add backup agent platform to the backup manager."""
        if not hasattr(platform, 'async_get_backup_agents'):
            return
        self.backup_agent_platforms[integration_domain] = platform

        @callback
        def listener() -> None:
            LOGGER.debug('Loading backup agents for %s', integration_domain)
            self.hass.async_create_task(self._async_reload_backup_agents(integration_domain))
        if hasattr(platform, 'async_register_backup_agents_listener'):
            platform.async_register_backup_agents_listener(self.hass, listener=listener)
        listener()

    async def _async_reload_backup_agents(self, domain: str) -> None:
        """Add backup agent platform to the backup manager."""
        platform = self.backup_agent_platforms[domain]
        for agent_id in list(self.backup_agents):
            if self.backup_agents[agent_id].domain == domain:
                del self.backup_agents[agent_id]
        for agent_id in list(self.local_backup_agents):
            if self.local_backup_agents[agent_id].domain == domain:
                del self.local_backup_agents[agent_id]
        agents = await platform.async_get_backup_agents(self.hass)
        self.backup_agents.update({agent.agent_id: agent for agent in agents})
        self.local_backup_agents.update({agent.agent_id: agent for agent in agents if isinstance(agent, LocalBackupAgent)})

    async def _add_platform(self, hass: HomeAssistant, integration_domain: str, platform: Any) -> None:
        """Add a backup platform manager."""
        self._add_platform_pre_post_handler(integration_domain, platform)
        self._async_add_backup_agent_platform(integration_domain, platform)
        LOGGER.debug('Backup platform %s loaded', integration_domain)
        LOGGER.debug('%s platforms loaded in total', len(self.platforms))
        LOGGER.debug('%s agents loaded in total', len(self.backup_agents))
        LOGGER.debug('%s local agents loaded in total', len(self.local_backup_agents))

    async def async_pre_backup_actions(self) -> None:
        """Perform pre backup actions."""
        pre_backup_results = await asyncio.gather(*(platform.async_pre_backup(self.hass) for platform in self.platforms.values()), return_exceptions=True)
        for result in pre_backup_results:
            if isinstance(result, Exception):
                raise BackupManagerError(f'Error during pre-backup: {result}') from result

    async def async_post_backup_actions(self) -> None:
        """Perform post backup actions."""
        post_backup_results = await asyncio.gather(*(platform.async_post_backup(self.hass) for platform in self.platforms.values()), return_exceptions=True)
        for result in post_backup_results:
            if isinstance(result, Exception):
                raise BackupManagerError(f'Error during post-backup: {result}') from result

    async def load_platforms(self) -> None:
        """Load backup platforms."""
        await integration_platform.async_process_integration_platforms(self.hass, DOMAIN, self._add_platform, wait_for_platforms=True)
        LOGGER.debug('Loaded %s platforms', len(self.platforms))
        LOGGER.debug('Loaded %s agents', len(self.backup_agents))

    async def _async_upload_backup(
        self,
        *,
        backup: AgentBackup,
        agent_ids: List[str],
        open_stream: Callable[[], Awaitable[AsyncIterator[bytes]]],
        password: Optional[str]
    ) -> Dict[str, Exception]:
        """Upload a backup to selected agents."""
        agent_errors: Dict[str, Exception] = {}
        LOGGER.debug('Uploading backup %s to agents %s', backup.backup_id, agent_ids)

        async def upload_backup_to_agent(agent_id: str) -> None:
            """Upload backup to a single agent, and encrypt or decrypt as needed."""
            config = self.config.data.agents.get(agent_id)
            should_encrypt = config.protected if config else password is not None
            streamer = None
            if should_encrypt == backup.protected or password is None:
                LOGGER.debug('Uploading backup %s to agent %s as is', backup.backup_id, agent_id)
                open_stream_func = open_stream
                _backup = backup
            elif should_encrypt:
                LOGGER.debug('Uploading encrypted backup %s to agent %s', backup.backup_id, agent_id)
                streamer = EncryptedBackupStreamer(self.hass, backup, open_stream, password)
            else:
                LOGGER.debug('Uploading decrypted backup %s to agent %s', backup.backup_id, agent_id)
                streamer = DecryptedBackupStreamer(self.hass, backup, open_stream, password)
            if streamer:
                open_stream_func = streamer.open_stream
                _backup = replace(backup, protected=should_encrypt, size=streamer.size())
            await self.backup_agents[agent_id].async_upload_backup(open_stream=open_stream_func, backup=_backup)
            if streamer:
                await streamer.wait()
        sync_backup_results = await asyncio.gather(*(upload_backup_to_agent(agent_id) for agent_id in agent_ids), return_exceptions=True)
        for idx, result in enumerate(sync_backup_results):
            agent_id = agent_ids[idx]
            if isinstance(result, BackupReaderWriterError):
                raise BackupManagerError(str(result)) from result
            if isinstance(result, BackupAgentError):
                agent_errors[agent_id] = result
                LOGGER.error('Upload failed for %s: %s', agent_id, result)
                continue
            if isinstance(result, Exception):
                agent_errors[agent_id] = result
                LOGGER.error('Unexpected error for %s: %s', agent_id, result, exc_info=result)
                continue
            if isinstance(result, BaseException):
                raise result
        return agent_errors

    async def async_get_backups(self) -> Tuple[Dict[str, ManagerBackup], Dict[str, Exception]]:
        """Get backups.

        Return a dictionary of Backup instances keyed by their ID.
        """
        backups: Dict[str, ManagerBackup] = {}
        agent_errors: Dict[str, Exception] = {}
        agent_ids = list(self.backup_agents)
        list_backups_results = await asyncio.gather(*(agent.async_list_backups() for agent in self.backup_agents.values()), return_exceptions=True)
        for idx, result