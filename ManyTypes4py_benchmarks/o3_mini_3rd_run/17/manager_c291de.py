from __future__ import annotations
import abc
import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
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
from typing import Any, AsyncGenerator, Coroutine, Dict, List, Optional, Tuple, TypeVar, Union, cast

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

# Type alias for backup event subscriber callback.
BackupEventCallback = Callable[[ManagerStateEvent], None]

@dataclass(frozen=True, kw_only=True, slots=True)
class NewBackup:
    backup_job_id: str

@dataclass(frozen=True, kw_only=True, slots=True)
class AgentBackupStatus:
    protected: bool
    size: int

@dataclass(frozen=True, kw_only=True, slots=True)
class ManagerBackup(BaseBackup):
    agents: Dict[str, AgentBackupStatus]
    addons: List[Any]
    backup_id: str
    date: str
    database_included: bool
    extra_metadata: Dict[str, Any]
    failed_agent_ids: List[str]
    folders: List[Any]
    homeassistant_included: bool
    homeassistant_version: str
    name: str
    with_automatic_settings: Optional[bool]

@dataclass(frozen=True, kw_only=True, slots=True)
class WrittenBackup:
    backup: BaseBackup
    open_stream: Callable[[], Coroutine[Any, Any, AsyncIterator[bytes]]]
    release_stream: Callable[[], Coroutine[Any, Any, None]]

class BackupManagerState(StrEnum):
    IDLE = 'idle'
    CREATE_BACKUP = 'create_backup'
    RECEIVE_BACKUP = 'receive_backup'
    RESTORE_BACKUP = 'restore_backup'

class CreateBackupStage(StrEnum):
    ADDON_REPOSITORIES = 'addon_repositories'
    ADDONS = 'addons'
    AWAIT_ADDON_RESTARTS = 'await_addon_restarts'
    DOCKER_CONFIG = 'docker_config'
    FINISHING_FILE = 'finishing_file'
    FOLDERS = 'folders'
    HOME_ASSISTANT = 'home_assistant'
    UPLOAD_TO_AGENTS = 'upload_to_agents'

class CreateBackupState(StrEnum):
    COMPLETED = 'completed'
    FAILED = 'failed'
    IN_PROGRESS = 'in_progress'

class ReceiveBackupStage(StrEnum):
    RECEIVE_FILE = 'receive_file'
    UPLOAD_TO_AGENTS = 'upload_to_agents'

class ReceiveBackupState(StrEnum):
    COMPLETED = 'completed'
    FAILED = 'failed'
    IN_PROGRESS = 'in_progress'

class RestoreBackupStage(StrEnum):
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
    COMPLETED = 'completed'
    CORE_RESTART = 'core_restart'
    FAILED = 'failed'
    IN_PROGRESS = 'in_progress'

@dataclass(frozen=True, kw_only=True, slots=True)
class ManagerStateEvent:
    reason: Optional[str] = None
    stage: Optional[Any] = None
    state: Optional[Any] = None

@dataclass(frozen=True, kw_only=True, slots=True)
class IdleEvent(ManagerStateEvent):
    manager_state: BackupManagerState = BackupManagerState.IDLE

@dataclass(frozen=True, kw_only=True, slots=True)
class CreateBackupEvent(ManagerStateEvent):
    manager_state: BackupManagerState = BackupManagerState.CREATE_BACKUP

@dataclass(frozen=True, kw_only=True, slots=True)
class ReceiveBackupEvent(ManagerStateEvent):
    manager_state: BackupManagerState = BackupManagerState.RECEIVE_BACKUP

@dataclass(frozen=True, kw_only=True, slots=True)
class RestoreBackupEvent(ManagerStateEvent):
    manager_state: BackupManagerState = BackupManagerState.RESTORE_BACKUP

class BackupPlatformProtocol(abc.ABC):
    async def async_pre_backup(self, hass: HomeAssistant) -> None:
        ...

    async def async_post_backup(self, hass: HomeAssistant) -> None:
        ...

class BackupReaderWriter(abc.ABC):
    @abc.abstractmethod
    async def async_create_backup(
        self,
        *,
        agent_ids: List[str],
        backup_name: str,
        extra_metadata: Dict[str, Any],
        include_addons: bool,
        include_all_addons: bool,
        include_database: bool,
        include_folders: bool,
        include_homeassistant: bool,
        on_progress: Callable[[ManagerStateEvent], None],
        password: Optional[str]
    ) -> Tuple[NewBackup, asyncio.Task]:
        """Create a backup."""
        ...

    @abc.abstractmethod
    async def async_receive_backup(
        self,
        *,
        agent_ids: List[str],
        stream: Any,
        suggested_filename: str
    ) -> WrittenBackup:
        """Receive a backup."""
        ...

    @abc.abstractmethod
    async def async_restore_backup(
        self,
        backup_id: str,
        *,
        agent_id: str,
        on_progress: Callable[[ManagerStateEvent], None],
        open_stream: Callable[[], Coroutine[Any, Any, AsyncIterator[bytes]]],
        password: Optional[str],
        restore_addons: bool,
        restore_database: bool,
        restore_folders: bool,
        restore_homeassistant: bool
    ) -> None:
        """Restore a backup."""
        ...

    @abc.abstractmethod
    async def async_resume_restore_progress_after_restart(
        self,
        *,
        on_progress: Callable[[ManagerStateEvent], None]
    ) -> None:
        """Get restore events after core restart."""
        ...

class IncorrectPasswordError(BackupReaderWriterError):
    error_code = 'password_incorrect'
    _message = 'The password provided is incorrect.'

class DecryptOnDowloadNotSupported(BackupManagerError):
    error_code = 'decrypt_on_download_not_supported'
    _message = 'On-the-fly decryption is not supported for this backup.'

class BackupManager:
    def __init__(self, hass: HomeAssistant, reader_writer: BackupReaderWriter) -> None:
        self.hass: HomeAssistant = hass
        self.platforms: Dict[str, Any] = {}
        self.backup_agent_platforms: Dict[str, Any] = {}
        self.backup_agents: Dict[str, BackupAgent] = {}
        self.local_backup_agents: Dict[str, LocalBackupAgent] = {}
        self.config: BackupConfig = BackupConfig(hass, self)
        self._reader_writer: BackupReaderWriter = reader_writer
        self.known_backups: KnownBackups = KnownBackups(self)
        self.store: BackupStore = BackupStore(hass, self)
        self._backup_task: Optional[asyncio.Task] = None
        self._backup_finish_task: Optional[asyncio.Task] = None
        self.remove_next_backup_event: Any = None
        self.remove_next_delete_event: Any = None
        self.last_event: ManagerStateEvent = IdleEvent()
        self.last_non_idle_event: Optional[ManagerStateEvent] = None
        self._backup_event_subscriptions: List[BackupEventCallback] = []

    async def async_setup(self) -> None:
        stored: Optional[Dict[str, Any]] = await self.store.load()
        if stored:
            self.config.load(stored['config'])
            self.known_backups.load(stored['backups'])
        await self._reader_writer.async_resume_restore_progress_after_restart(on_progress=self.async_on_backup_event)
        await self.load_platforms()

    @property
    def state(self) -> BackupManagerState:
        return self.last_event.manager_state

    @callback
    def _add_platform_pre_post_handler(self, integration_domain: str, platform: Any) -> None:
        if not hasattr(platform, 'async_pre_backup') or not hasattr(platform, 'async_post_backup'):
            return
        self.platforms[integration_domain] = platform

    @callback
    def _async_add_backup_agent_platform(self, integration_domain: str, platform: Any) -> None:
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
        platform: Any = self.backup_agent_platforms[domain]
        for agent_id in list(self.backup_agents):
            if self.backup_agents[agent_id].domain == domain:
                del self.backup_agents[agent_id]
        for agent_id in list(self.local_backup_agents):
            if self.local_backup_agents[agent_id].domain == domain:
                del self.local_backup_agents[agent_id]
        agents: List[BackupAgent] = await platform.async_get_backup_agents(self.hass)
        self.backup_agents.update({agent.agent_id: agent for agent in agents})
        self.local_backup_agents.update({agent.agent_id: agent for agent in agents if isinstance(agent, LocalBackupAgent)})

    async def _add_platform(self, hass: HomeAssistant, integration_domain: str, platform: Any) -> None:
        self._add_platform_pre_post_handler(integration_domain, platform)
        self._async_add_backup_agent_platform(integration_domain, platform)
        LOGGER.debug('Backup platform %s loaded', integration_domain)
        LOGGER.debug('%s platforms loaded in total', len(self.platforms))
        LOGGER.debug('%s agents loaded in total', len(self.backup_agents))
        LOGGER.debug('%s local agents loaded in total', len(self.local_backup_agents))

    async def async_pre_backup_actions(self) -> None:
        pre_backup_results = await asyncio.gather(
            *(platform.async_pre_backup(self.hass) for platform in self.platforms.values()),
            return_exceptions=True
        )
        for result in pre_backup_results:
            if isinstance(result, Exception):
                raise BackupManagerError(f'Error during pre-backup: {result}') from result

    async def async_post_backup_actions(self) -> None:
        post_backup_results = await asyncio.gather(
            *(platform.async_post_backup(self.hass) for platform in self.platforms.values()),
            return_exceptions=True
        )
        for result in post_backup_results:
            if isinstance(result, Exception):
                raise BackupManagerError(f'Error during post-backup: {result}') from result

    async def load_platforms(self) -> None:
        await integration_platform.async_process_integration_platforms(self.hass, DOMAIN, self._add_platform, wait_for_platforms=True)
        LOGGER.debug('Loaded %s platforms', len(self.platforms))
        LOGGER.debug('Loaded %s agents', len(self.backup_agents))

    async def _async_upload_backup(
        self,
        *,
        backup: BaseBackup,
        agent_ids: List[str],
        open_stream: Callable[[], Coroutine[Any, Any, AsyncIterator[bytes]]],
        password: Optional[str]
    ) -> Dict[str, Exception]:
        agent_errors: Dict[str, Exception] = {}
        LOGGER.debug('Uploading backup %s to agents %s', backup.backup_id, agent_ids)

        async def upload_backup_to_agent(agent_id: str) -> None:
            config = self.config.data.agents.get(agent_id)
            should_encrypt: bool = config.protected if config else (password is not None)
            streamer: Optional[Union[EncryptedBackupStreamer, DecryptedBackupStreamer]] = None
            if should_encrypt == backup.protected or password is None:
                LOGGER.debug('Uploading backup %s to agent %s as is', backup.backup_id, agent_id)
                open_stream_func = open_stream
                _backup = backup
            elif should_encrypt:
                LOGGER.debug('Uploading encrypted backup %s to agent %s', backup.backup_id, agent_id)
                streamer = EncryptedBackupStreamer(self.hass, backup, open_stream, password)
                open_stream_func = streamer.open_stream
                _backup = replace(backup, protected=should_encrypt, size=streamer.size())
            else:
                LOGGER.debug('Uploading decrypted backup %s to agent %s', backup.backup_id, agent_id)
                streamer = DecryptedBackupStreamer(self.hass, backup, open_stream, password)
                open_stream_func = streamer.open_stream
                _backup = replace(backup, protected=should_encrypt, size=streamer.size())
            await self.backup_agents[agent_id].async_upload_backup(open_stream=open_stream_func, backup=_backup)
            if streamer:
                await streamer.wait()

        upload_results = await asyncio.gather(
            *(upload_backup_to_agent(agent_id) for agent_id in agent_ids),
            return_exceptions=True
        )
        for idx, result in enumerate(upload_results):
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
        backups: Dict[str, ManagerBackup] = {}
        agent_errors: Dict[str, Exception] = {}
        agent_ids: List[str] = list(self.backup_agents)
        list_backups_results = await asyncio.gather(
            *(agent.async_list_backups() for agent in self.backup_agents.values()),
            return_exceptions=True
        )
        for idx, result in enumerate(list_backups_results):
            agent_id = agent_ids[idx]
            if isinstance(result, BackupAgentError):
                agent_errors[agent_id] = result
                continue
            if isinstance(result, Exception):
                agent_errors[agent_id] = result
                LOGGER.error('Unexpected error for %s: %s', agent_id, result, exc_info=result)
                continue
            if isinstance(result, BaseException):
                raise result
            for agent_backup in result:
                backup_id: str = agent_backup.backup_id
                if backup_id not in backups:
                    if (known_backup := self.known_backups.get(backup_id)):
                        failed_agent_ids: List[str] = known_backup.failed_agent_ids
                    else:
                        failed_agent_ids = []
                    instance_id_str: str = await instance_id.async_get(self.hass)
                    with_automatic_settings: Optional[bool] = self.is_our_automatic_backup(agent_backup, instance_id_str)
                    backups[backup_id] = ManagerBackup(
                        agents={},
                        addons=agent_backup.addons,
                        backup_id=backup_id,
                        date=agent_backup.date,
                        database_included=agent_backup.database_included,
                        extra_metadata=agent_backup.extra_metadata,
                        failed_agent_ids=failed_agent_ids,
                        folders=agent_backup.folders,
                        homeassistant_included=agent_backup.homeassistant_included,
                        homeassistant_version=agent_backup.homeassistant_version,
                        name=agent_backup.name,
                        with_automatic_settings=with_automatic_settings
                    )
                backups[backup_id].agents[agent_id] = AgentBackupStatus(
                    protected=agent_backup.protected,
                    size=agent_backup.size
                )
        return backups, agent_errors

    async def async_get_backup(self, backup_id: str) -> Tuple[Optional[ManagerBackup], Dict[str, Exception]]:
        backup: Optional[ManagerBackup] = None
        agent_errors: Dict[str, Exception] = {}
        agent_ids: List[str] = list(self.backup_agents)
        get_backup_results = await asyncio.gather(
            *(agent.async_get_backup(backup_id) for agent in self.backup_agents.values()),
            return_exceptions=True
        )
        for idx, result in enumerate(get_backup_results):
            agent_id = agent_ids[idx]
            if isinstance(result, BackupAgentError):
                agent_errors[agent_id] = result
                continue
            if isinstance(result, Exception):
                agent_errors[agent_id] = result
                LOGGER.error('Unexpected error for %s: %s', agent_id, result, exc_info=result)
                continue
            if isinstance(result, BaseException):
                raise result
            if not result:
                continue
            if backup is None:
                if (known_backup := self.known_backups.get(backup_id)):
                    failed_agent_ids: List[str] = known_backup.failed_agent_ids
                else:
                    failed_agent_ids = []
                instance_id_str: str = await instance_id.async_get(self.hass)
                with_automatic_settings: Optional[bool] = self.is_our_automatic_backup(result, instance_id_str)
                backup = ManagerBackup(
                    agents={},
                    addons=result.addons,
                    backup_id=result.backup_id,
                    date=result.date,
                    database_included=result.database_included,
                    extra_metadata=result.extra_metadata,
                    failed_agent_ids=failed_agent_ids,
                    folders=result.folders,
                    homeassistant_included=result.homeassistant_included,
                    homeassistant_version=result.homeassistant_version,
                    name=result.name,
                    with_automatic_settings=with_automatic_settings
                )
            backup.agents[agent_id] = AgentBackupStatus(
                protected=result.protected,
                size=result.size
            )
        return backup, agent_errors

    @staticmethod
    def is_our_automatic_backup(backup: BaseBackup, our_instance_id: str) -> Optional[bool]:
        if backup.extra_metadata.get('instance_id') != our_instance_id:
            return None
        with_automatic_settings = backup.extra_metadata.get('with_automatic_settings')
        if not isinstance(with_automatic_settings, bool):
            return None
        return with_automatic_settings

    async def async_delete_backup(self, backup_id: str, *, agent_ids: Optional[List[str]] = None) -> Dict[str, Exception]:
        agent_errors: Dict[str, Exception] = {}
        if agent_ids is None:
            agent_ids = list(self.backup_agents)
        delete_backup_results = await asyncio.gather(
            *(self.backup_agents[agent_id].async_delete_backup(backup_id) for agent_id in agent_ids),
            return_exceptions=True
        )
        for idx, result in enumerate(delete_backup_results):
            agent_id = agent_ids[idx]
            if isinstance(result, BackupAgentError):
                agent_errors[agent_id] = result
                continue
            if isinstance(result, Exception):
                agent_errors[agent_id] = result
                LOGGER.error('Unexpected error for %s: %s', agent_id, result, exc_info=result)
                continue
            if isinstance(result, BaseException):
                raise result
        if not agent_errors:
            self.known_backups.remove(backup_id)
        return agent_errors

    async def async_delete_filtered_backups(
        self,
        *,
        include_filter: Callable[[Dict[str, ManagerBackup]], Dict[str, ManagerBackup]],
        delete_filter: Callable[[Dict[str, ManagerBackup]], Dict[str, ManagerBackup]]
    ) -> None:
        backups, get_agent_errors = await self.async_get_backups()
        if get_agent_errors:
            LOGGER.debug('Error getting backups; continuing anyway: %s', get_agent_errors)
        backups = include_filter(backups)
        backups_by_agent: Dict[str, Dict[str, ManagerBackup]] = defaultdict(dict)
        for backup_id, backup in backups.items():
            for agent_id in backup.agents:
                backups_by_agent[agent_id][backup_id] = backup
        LOGGER.debug('Backups returned by include filter: %s', backups)
        LOGGER.debug('Backups returned by include filter by agent: %s', {agent_id: list(backups) for agent_id, backups in backups_by_agent.items()})
        backups_to_delete = delete_filter(backups)
        LOGGER.debug('Backups returned by delete filter: %s', backups_to_delete)
        if not backups_to_delete:
            return
        backups_to_delete_by_agent: Dict[str, Dict[str, ManagerBackup]] = defaultdict(dict)
        for backup_id, backup in sorted(backups_to_delete.items(), key=lambda backup_item: backup_item[1].date):
            for agent_id in backup.agents:
                backups_to_delete_by_agent[agent_id][backup_id] = backup
        LOGGER.debug('Backups returned by delete filter by agent: %s', {agent_id: list(backups) for agent_id, backups in backups_to_delete_by_agent.items()})
        for agent_id, to_delete_from_agent in backups_to_delete_by_agent.items():
            if len(to_delete_from_agent) >= len(backups_by_agent[agent_id]):
                last_backup = to_delete_from_agent.popitem()
                LOGGER.debug('Keeping the last backup %s for agent %s', last_backup, agent_id)
        LOGGER.debug('Backups to delete by agent: %s', {agent_id: list(backups) for agent_id, backups in backups_to_delete_by_agent.items()})
        backup_ids_to_delete: Dict[str, set[str]] = defaultdict(set)
        for agent_id, to_delete in backups_to_delete_by_agent.items():
            for backup_id in to_delete:
                backup_ids_to_delete[backup_id].add(agent_id)
        if not backup_ids_to_delete:
            return
        backup_ids = list(backup_ids_to_delete)
        delete_results = await asyncio.gather(
            *(self.async_delete_backup(backup_id, agent_ids=list(agent_ids))
              for backup_id, agent_ids in backup_ids_to_delete.items())
        )
        agent_errors = {backup_id: error for backup_id, error in zip(backup_ids, delete_results, strict=True) if error}
        if agent_errors:
            LOGGER.error('Error deleting old copies: %s', agent_errors)

    async def async_receive_backup(self, *, agent_ids: List[str], contents: Any) -> str:
        if self.state is not BackupManagerState.IDLE:
            raise BackupManagerError(f'Backup manager busy: {self.state}')
        self.async_on_backup_event(ReceiveBackupEvent(reason=None, stage=None, state=ReceiveBackupState.IN_PROGRESS))
        try:
            backup_id: str = await self._async_receive_backup(agent_ids=agent_ids, contents=contents)
        except Exception:
            self.async_on_backup_event(ReceiveBackupEvent(reason='unknown_error', stage=None, state=ReceiveBackupState.FAILED))
            raise
        else:
            self.async_on_backup_event(ReceiveBackupEvent(reason=None, stage=None, state=ReceiveBackupState.COMPLETED))
            return backup_id
        finally:
            self.async_on_backup_event(IdleEvent())

    async def _async_receive_backup(self, *, agent_ids: List[str], contents: Any) -> str:
        contents.chunk_size = BUF_SIZE
        self.async_on_backup_event(ReceiveBackupEvent(reason=None, stage=ReceiveBackupStage.RECEIVE_FILE, state=ReceiveBackupState.IN_PROGRESS))
        written_backup: WrittenBackup = await self._reader_writer.async_receive_backup(agent_ids=agent_ids, stream=contents, suggested_filename=getattr(contents, "filename", "backup.tar"))
        self.async_on_backup_event(ReceiveBackupEvent(reason=None, stage=ReceiveBackupStage.UPLOAD_TO_AGENTS, state=ReceiveBackupState.IN_PROGRESS))
        agent_errors: Dict[str, Exception] = await self._async_upload_backup(backup=written_backup.backup, agent_ids=agent_ids, open_stream=written_backup.open_stream, password=None)
        await written_backup.release_stream()
        self.known_backups.add(written_backup.backup, agent_errors, [])
        return written_backup.backup.backup_id

    async def async_create_backup(
        self,
        *,
        agent_ids: List[str],
        extra_metadata: Optional[Dict[str, Any]],
        include_addons: bool,
        include_all_addons: bool,
        include_database: bool,
        include_folders: bool,
        include_homeassistant: bool,
        name: str,
        password: Optional[str],
        with_automatic_settings: bool = False
    ) -> NewBackup:
        new_backup: NewBackup = await self.async_initiate_backup(
            agent_ids=agent_ids,
            extra_metadata=extra_metadata,
            include_addons=include_addons,
            include_all_addons=include_all_addons,
            include_database=include_database,
            include_folders=include_folders,
            include_homeassistant=include_homeassistant,
            name=name,
            password=password,
            raise_task_error=True,
            with_automatic_settings=with_automatic_settings
        )
        assert self._backup_finish_task is not None
        await self._backup_finish_task
        return new_backup

    async def async_create_automatic_backup(self) -> NewBackup:
        config_data = self.config.data
        return await self.async_create_backup(
            agent_ids=config_data.create_backup.agent_ids,
            include_addons=config_data.create_backup.include_addons,
            include_all_addons=config_data.create_backup.include_all_addons,
            include_database=config_data.create_backup.include_database,
            include_folders=config_data.create_backup.include_folders,
            include_homeassistant=True,
            name=config_data.create_backup.name,
            password=config_data.create_backup.password,
            with_automatic_settings=True
        )

    async def async_initiate_backup(
        self,
        *,
        agent_ids: List[str],
        extra_metadata: Optional[Dict[str, Any]],
        include_addons: bool,
        include_all_addons: bool,
        include_database: bool,
        include_folders: bool,
        include_homeassistant: bool,
        name: str,
        password: Optional[str],
        raise_task_error: bool = False,
        with_automatic_settings: bool = False
    ) -> NewBackup:
        if self.state is not BackupManagerState.IDLE:
            raise BackupManagerError(f'Backup manager busy: {self.state}')
        if with_automatic_settings:
            self.config.data.last_attempted_automatic_backup = dt_util.now()
            self.store.save()
        self.async_on_backup_event(CreateBackupEvent(reason=None, stage=None, state=CreateBackupState.IN_PROGRESS))
        try:
            return await self._async_create_backup(
                agent_ids=agent_ids,
                extra_metadata=extra_metadata,
                include_addons=include_addons,
                include_all_addons=include_all_addons,
                include_database=include_database,
                include_folders=include_folders,
                include_homeassistant=include_homeassistant,
                name=name,
                password=password,
                raise_task_error=raise_task_error,
                with_automatic_settings=with_automatic_settings
            )
        except Exception as err:
            reason: str = err.error_code if isinstance(err, BackupError) else 'unknown_error'
            self.async_on_backup_event(CreateBackupEvent(reason=reason, stage=None, state=CreateBackupState.FAILED))
            self.async_on_backup_event(IdleEvent())
            if with_automatic_settings:
                self._update_issue_backup_failed()
            raise

    async def _async_create_backup(
        self,
        *,
        agent_ids: List[str],
        extra_metadata: Optional[Dict[str, Any]],
        include_addons: bool,
        include_all_addons: bool,
        include_database: bool,
        include_folders: bool,
        include_homeassistant: bool,
        name: str,
        password: Optional[str],
        raise_task_error: bool,
        with_automatic_settings: bool
    ) -> NewBackup:
        unavailable_agents: List[str] = [agent_id for agent_id in agent_ids if agent_id not in self.backup_agents]
        available_agents: List[str] = [agent_id for agent_id in agent_ids if agent_id in self.backup_agents]
        if not available_agents:
            raise BackupManagerError(f'At least one available backup agent must be selected, got {agent_ids}')
        if unavailable_agents:
            LOGGER.warning('Backup agents %s are not available, will backupp to %s', unavailable_agents, available_agents)
        if include_all_addons and include_addons:
            raise BackupManagerError('Cannot include all addons and specify specific addons')
        backup_name: str = (name if name is None else name.strip()) or f'{("Automatic" if with_automatic_settings else "Custom")} backup {HAVERSION}'
        extra_metadata = extra_metadata or {}
        try:
            new_backup, self._backup_task = await self._reader_writer.async_create_backup(
                agent_ids=available_agents,
                backup_name=backup_name,
                extra_metadata={**extra_metadata, 'instance_id': await instance_id.async_get(self.hass), 'with_automatic_settings': with_automatic_settings},
                include_addons=include_addons,
                include_all_addons=include_all_addons,
                include_database=include_database,
                include_folders=include_folders,
                include_homeassistant=include_homeassistant,
                on_progress=self.async_on_backup_event,
                password=password
            )
        except BackupReaderWriterError as err:
            raise BackupManagerError(str(err)) from err
        backup_finish_task: asyncio.Task = self._backup_finish_task = self.hass.async_create_task(
            self._async_finish_backup(available_agents, unavailable_agents, with_automatic_settings, password),
            name='backup_manager_finish_backup'
        )
        if not raise_task_error:
            def log_finish_task_error(task: asyncio.Task) -> None:
                if task.done() and (not task.cancelled()) and (err := task.exception()):
                    if isinstance(err, BackupManagerError):
                        LOGGER.error('Error creating backup: %s', err)
                    else:
                        LOGGER.error('Unexpected error: %s', err, exc_info=err)
            backup_finish_task.add_done_callback(log_finish_task_error)
        return new_backup

    async def _async_finish_backup(
        self,
        available_agents: List[str],
        unavailable_agents: List[str],
        with_automatic_settings: bool,
        password: Optional[str]
    ) -> None:
        backup_success: bool = False
        try:
            if self._backup_task is None:
                raise BackupManagerError("Missing backup task")
            written_backup: WrittenBackup = await self._backup_task
        except Exception as err:
            if with_automatic_settings:
                self._update_issue_backup_failed()
            if isinstance(err, BackupReaderWriterError):
                raise BackupManagerError(str(err)) from err
            raise
        else:
            LOGGER.debug('Generated new backup with backup_id %s, uploading to agents %s', written_backup.backup.backup_id, available_agents)
            self.async_on_backup_event(CreateBackupEvent(reason=None, stage=CreateBackupStage.UPLOAD_TO_AGENTS, state=CreateBackupState.IN_PROGRESS))
            try:
                agent_errors: Dict[str, Exception] = await self._async_upload_backup(
                    backup=written_backup.backup,
                    agent_ids=available_agents,
                    open_stream=written_backup.open_stream,
                    password=password
                )
            finally:
                await written_backup.release_stream()
            self.known_backups.add(written_backup.backup, agent_errors, unavailable_agents)
            if not agent_errors:
                if with_automatic_settings:
                    self.config.data.last_completed_automatic_backup = dt_util.now()
                    self.store.save()
                backup_success = True
            if with_automatic_settings:
                self._update_issue_after_agent_upload(agent_errors, unavailable_agents)
            await delete_backups_exceeding_configured_count(self)
        finally:
            self._backup_task = None
            self._backup_finish_task = None
            if backup_success:
                self.async_on_backup_event(CreateBackupEvent(reason=None, stage=None, state=CreateBackupState.COMPLETED))
            else:
                self.async_on_backup_event(CreateBackupEvent(reason='upload_failed', stage=None, state=CreateBackupState.FAILED))
            self.async_on_backup_event(IdleEvent())

    async def async_restore_backup(
        self,
        backup_id: str,
        *,
        agent_id: str,
        password: Optional[str],
        restore_addons: bool,
        restore_database: bool,
        restore_folders: bool,
        restore_homeassistant: bool
    ) -> None:
        if self.state is not BackupManagerState.IDLE:
            raise BackupManagerError(f'Backup manager busy: {self.state}')
        self.async_on_backup_event(RestoreBackupEvent(reason=None, stage=None, state=RestoreBackupState.IN_PROGRESS))
        try:
            await self._async_restore_backup(
                backup_id=backup_id,
                agent_id=agent_id,
                password=password,
                restore_addons=restore_addons,
                restore_database=restore_database,
                restore_folders=restore_folders,
                restore_homeassistant=restore_homeassistant
            )
            self.async_on_backup_event(RestoreBackupEvent(reason=None, stage=None, state=RestoreBackupState.COMPLETED))
        except BackupError as err:
            self.async_on_backup_event(RestoreBackupEvent(reason=err.error_code, stage=None, state=RestoreBackupState.FAILED))
            raise
        except Exception:
            self.async_on_backup_event(RestoreBackupEvent(reason='unknown_error', stage=None, state=RestoreBackupState.FAILED))
            raise
        finally:
            self.async_on_backup_event(IdleEvent())

    async def _async_restore_backup(
        self,
        backup_id: str,
        *,
        agent_id: str,
        password: Optional[str],
        restore_addons: bool,
        restore_database: bool,
        restore_folders: bool,
        restore_homeassistant: bool
    ) -> None:
        agent: BackupAgent = self.backup_agents[agent_id]
        if not await agent.async_get_backup(backup_id):
            raise BackupManagerError(f'Backup {backup_id} not found in agent {agent_id}')

        async def open_backup() -> AsyncIterator[bytes]:
            return await agent.async_download_backup(backup_id)
        await self._reader_writer.async_restore_backup(
            backup_id=backup_id,
            open_stream=open_backup,
            agent_id=agent_id,
            on_progress=self.async_on_backup_event,
            password=password,
            restore_addons=restore_addons,
            restore_database=restore_database,
            restore_folders=restore_folders,
            restore_homeassistant=restore_homeassistant
        )

    @callback
    def async_on_backup_event(self, event: ManagerStateEvent) -> None:
        if (current_state := self.state) != (new_state := event.manager_state):
            LOGGER.debug('Backup state: %s -> %s', current_state, new_state)
        self.last_event = event
        if not isinstance(event, IdleEvent):
            self.last_non_idle_event = event
        for subscription in self._backup_event_subscriptions:
            subscription(event)

    @callback
    def async_subscribe_events(self, on_event: BackupEventCallback) -> Callable[[], None]:
        self._backup_event_subscriptions.append(on_event)
        def remove_subscription() -> None:
            self._backup_event_subscriptions.remove(on_event)
        return remove_subscription

    def _update_issue_backup_failed(self) -> None:
        ir.async_create_issue(
            self.hass,
            DOMAIN,
            'automatic_backup_failed',
            is_fixable=False,
            is_persistent=True,
            learn_more_url='homeassistant://config/backup',
            severity=ir.IssueSeverity.WARNING,
            translation_key='automatic_backup_failed_create'
        )

    def _update_issue_after_agent_upload(self, agent_errors: Dict[str, Exception], unavailable_agents: List[str]) -> None:
        if not agent_errors and (not unavailable_agents):
            ir.async_delete_issue(self.hass, DOMAIN, 'automatic_backup_failed')
            return
        failed_agents = ', '.join(chain((self.backup_agents[agent_id].name for agent_id in agent_errors), unavailable_agents))
        ir.async_create_issue(
            self.hass,
            DOMAIN,
            'automatic_backup_failed',
            is_fixable=False,
            is_persistent=True,
            learn_more_url='homeassistant://config/backup',
            severity=ir.IssueSeverity.WARNING,
            translation_key='automatic_backup_failed_upload_agents',
            translation_placeholders={'failed_agents': failed_agents}
        )

    async def async_can_decrypt_on_download(self, backup_id: str, *, agent_id: str, password: Optional[str]) -> None:
        try:
            agent: BackupAgent = self.backup_agents[agent_id]
        except KeyError as err:
            raise BackupManagerError(f'Invalid agent selected: {agent_id}') from err
        if not await agent.async_get_backup(backup_id):
            raise BackupManagerError(f'Backup {backup_id} not found in agent {agent_id}')
        if agent_id in self.local_backup_agents:
            local_agent: LocalBackupAgent = self.local_backup_agents[agent_id]
            path: Path = local_agent.get_backup_path(backup_id)
            reader = await self.hass.async_add_executor_job(open, path.as_posix(), 'rb')
        else:
            backup_stream = await agent.async_download_backup(backup_id)
            reader = cast(Any, AsyncIteratorReader(self.hass, backup_stream))
        try:
            await self.hass.async_add_executor_job(validate_password_stream, reader, password)
        except backup_util.IncorrectPassword as err:
            raise IncorrectPasswordError from err
        except backup_util.UnsupportedSecureTarVersion as err:
            raise DecryptOnDowloadNotSupported from err
        except backup_util.DecryptError as err:
            raise BackupManagerError(str(err)) from err
        finally:
            reader.close()

class KnownBackups:
    def __init__(self, manager: BackupManager) -> None:
        self._backups: Dict[str, KnownBackup] = {}
        self._manager: BackupManager = manager

    def load(self, stored_backups: Any) -> None:
        self._backups = {backup['backup_id']: KnownBackup(backup_id=backup['backup_id'], failed_agent_ids=backup['failed_agent_ids']) for backup in stored_backups}

    def to_list(self) -> List[Dict[str, Any]]:
        return [backup.to_dict() for backup in self._backups.values()]

    def add(self, backup: BaseBackup, agent_errors: Dict[str, Exception], unavailable_agents: List[str]) -> None:
        self._backups[backup.backup_id] = KnownBackup(backup_id=backup.backup_id, failed_agent_ids=list(chain(agent_errors, unavailable_agents)))
        self._manager.store.save()

    def get(self, backup_id: str) -> Optional[KnownBackup]:
        return self._backups.get(backup_id)

    def remove(self, backup_id: str) -> None:
        if backup_id not in self._backups:
            return
        self._backups.pop(backup_id)
        self._manager.store.save()

@dataclass(kw_only=True)
class KnownBackup:
    backup_id: str
    failed_agent_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {'backup_id': self.backup_id, 'failed_agent_ids': self.failed_agent_ids}

class StoredKnownBackup(TypedDict):
    backup_id: str
    failed_agent_ids: List[str]

class CoreBackupReaderWriter(BackupReaderWriter):
    _local_agent_id: str = f'{DOMAIN}.local'

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass: HomeAssistant = hass
        self.temp_backup_dir: Path = Path(hass.config.path('tmp_backups'))

    async def async_create_backup(
        self,
        *,
        agent_ids: List[str],
        backup_name: str,
        extra_metadata: Dict[str, Any],
        include_addons: bool,
        include_all_addons: bool,
        include_database: bool,
        include_folders: bool,
        include_homeassistant: bool,
        on_progress: Callable[[ManagerStateEvent], None],
        password: Optional[str]
    ) -> Tuple[NewBackup, asyncio.Task]:
        date_str: str = dt_util.now().isoformat()
        backup_id: str = _generate_backup_id(date_str, backup_name)
        if include_addons or include_all_addons or include_folders:
            raise BackupReaderWriterError('Addons and folders are not supported by core backup')
        if not include_homeassistant:
            raise BackupReaderWriterError('Home Assistant must be included in backup')
        backup_task: asyncio.Task = self._hass.async_create_task(
            self._async_create_backup(
                agent_ids=agent_ids,
                backup_id=backup_id,
                backup_name=backup_name,
                extra_metadata=extra_metadata,
                include_database=include_database,
                date_str=date_str,
                on_progress=on_progress,
                password=password
            ),
            name='backup_manager_create_backup',
            eager_start=False
        )
        return (NewBackup(backup_job_id=backup_id), backup_task)

    async def _async_create_backup(
        self,
        *,
        agent_ids: List[str],
        backup_id: str,
        backup_name: str,
        date_str: str,
        extra_metadata: Dict[str, Any],
        include_database: bool,
        on_progress: Callable[[ManagerStateEvent], None],
        password: Optional[str]
    ) -> WrittenBackup:
        manager: BackupManager = self._hass.data[DATA_MANAGER]
        agent_config = manager.config.data.agents.get(self._local_agent_id)
        if self._local_agent_id in agent_ids and agent_config and (not agent_config.protected):
            password = None
        backup = AgentBackup(
            addons=[],
            backup_id=backup_id,
            database_included=include_database,
            date=date_str,
            extra_metadata=extra_metadata,
            folders=[],
            homeassistant_included=True,
            homeassistant_version=HAVERSION,
            name=backup_name,
            protected=password is not None,
            size=0
        )
        local_agent_tar_file_path: Optional[Path] = None
        if self._local_agent_id in agent_ids:
            local_agent: LocalBackupAgent = manager.local_backup_agents[self._local_agent_id]
            local_agent_tar_file_path = local_agent.get_new_backup_path(backup)
        on_progress(CreateBackupEvent(reason=None, stage=CreateBackupStage.HOME_ASSISTANT, state=CreateBackupState.IN_PROGRESS))
        try:
            await manager.async_pre_backup_actions()
            backup_data: Dict[str, Any] = {
                'compressed': True,
                'date': date_str,
                'extra': extra_metadata,
                'homeassistant': {'exclude_database': not include_database, 'version': HAVERSION},
                'name': backup_name,
                'protected': password is not None,
                'slug': backup_id,
                'type': 'partial',
                'version': 2
            }
            tar_file_path, size_in_bytes = await self._hass.async_add_executor_job(
                self._mkdir_and_generate_backup_contents,
                backup_data,
                include_database,
                password,
                local_agent_tar_file_path
            )
        except (BackupManagerError, OSError, tarfile.TarError, ValueError) as err:
            raise BackupReaderWriterError(str(err)) from err
        else:
            backup = replace(backup, size=size_in_bytes)
            async_add_executor_job = self._hass.async_add_executor_job

            async def send_backup() -> AsyncGenerator[bytes, None]:
                f = await async_add_executor_job(tar_file_path.open, 'rb')
                try:
                    while (chunk := (await async_add_executor_job(f.read, 2 ** 20))):
                        yield chunk
                finally:
                    await async_add_executor_job(f.close)

            async def open_backup() -> AsyncIterator[bytes]:
                return send_backup()

            async def remove_backup() -> None:
                if local_agent_tar_file_path:
                    return
                try:
                    await async_add_executor_job(tar_file_path.unlink, True)
                except OSError as err:
                    raise BackupReaderWriterError(str(err)) from err
            return WrittenBackup(backup=backup, open_stream=open_backup, release_stream=remove_backup)
        finally:
            try:
                await manager.async_post_backup_actions()
            except BackupManagerError as err:
                raise BackupReaderWriterError(str(err)) from err

    def _mkdir_and_generate_backup_contents(self, backup_data: Dict[str, Any], database_included: bool, password: Optional[str], tar_file_path: Optional[Path]) -> Tuple[Path, int]:
        if not tar_file_path:
            tar_file_path = self.temp_backup_dir / f"{backup_data['slug']}.tar"
        make_backup_dir(tar_file_path.parent)
        excludes = EXCLUDE_FROM_BACKUP
        if not database_included:
            excludes = excludes + EXCLUDE_DATABASE_FROM_BACKUP

        def is_excluded_by_filter(path: Path) -> bool:
            for exclude in excludes:
                if not path.match(exclude):
                    continue
                LOGGER.debug('Ignoring %s because of %s', path, exclude)
                return True
            return False
        outer_secure_tarfile = SecureTarFile(tar_file_path, 'w', gzip=False, bufsize=BUF_SIZE)
        with outer_secure_tarfile as outer_secure_tarfile_tarfile:
            raw_bytes = json_bytes(backup_data)
            fileobj = io.BytesIO(raw_bytes)
            tar_info = tarfile.TarInfo(name='./backup.json')
            tar_info.size = len(raw_bytes)
            tar_info.mtime = int(time.time())
            outer_secure_tarfile_tarfile.addfile(tar_info, fileobj=fileobj)
            with outer_secure_tarfile.create_inner_tar('./homeassistant.tar.gz', gzip=True, key=password_to_key(password) if password is not None else None) as core_tar:
                atomic_contents_add(tar_file=core_tar, origin_path=Path(self._hass.config.path()), file_filter=is_excluded_by_filter, arcname='data')
        return tar_file_path, tar_file_path.stat().st_size

    async def async_receive_backup(
        self,
        *,
        agent_ids: List[str],
        stream: Any,
        suggested_filename: str
    ) -> WrittenBackup:
        temp_file: Path = Path(self.temp_backup_dir, suggested_filename)
        async_add_executor_job = self._hass.async_add_executor_job
        await async_add_executor_job(make_backup_dir, self.temp_backup_dir)
        f = await async_add_executor_job(temp_file.open, 'wb')
        try:
            async for chunk in stream:
                await async_add_executor_job(f.write, chunk)
        finally:
            await async_add_executor_job(f.close)
        try:
            backup = await async_add_executor_job(read_backup, temp_file)
        except (OSError, tarfile.TarError, json.JSONDecodeError, KeyError) as err:
            LOGGER.warning('Unable to parse backup %s: %s', temp_file, err)
            raise
        manager: BackupManager = self._hass.data[DATA_MANAGER]
        if self._local_agent_id in agent_ids:
            local_agent: LocalBackupAgent = manager.local_backup_agents[self._local_agent_id]
            tar_file_path: Path = local_agent.get_new_backup_path(backup)
            await async_add_executor_job(make_backup_dir, tar_file_path.parent)
            await async_add_executor_job(shutil.move, temp_file, tar_file_path)
        else:
            tar_file_path = temp_file

        async def send_backup() -> AsyncGenerator[bytes, None]:
            f_local = await async_add_executor_job(tar_file_path.open, 'rb')
            try:
                while (chunk := (await async_add_executor_job(f_local.read, 2 ** 20))):
                    yield chunk
            finally:
                await async_add_executor_job(f_local.close)

        async def open_backup() -> AsyncIterator[bytes]:
            return send_backup()

        async def remove_backup() -> None:
            if self._local_agent_id in agent_ids:
                return
            await async_add_executor_job(temp_file.unlink, True)
        return WrittenBackup(backup=backup, open_stream=open_backup, release_stream=remove_backup)

    async def async_restore_backup(
        self,
        backup_id: str,
        open_stream: Callable[[], Coroutine[Any, Any, AsyncIterator[bytes]]],
        *,
        agent_id: str,
        on_progress: Callable[[ManagerStateEvent], None],
        password: Optional[str],
        restore_addons: bool,
        restore_database: bool,
        restore_folders: bool,
        restore_homeassistant: bool
    ) -> None:
        if restore_addons or restore_folders:
            raise BackupReaderWriterError('Addons and folders are not supported in core restore')
        if not restore_homeassistant and (not restore_database):
            raise BackupReaderWriterError('Home Assistant or database must be included in restore')
        manager: BackupManager = self._hass.data[DATA_MANAGER]
        if agent_id in manager.local_backup_agents:
            local_agent: LocalBackupAgent = manager.local_backup_agents[agent_id]
            path: Path = local_agent.get_backup_path(backup_id)
            remove_after_restore: bool = False
        else:
            async_add_executor_job = self._hass.async_add_executor_job
            path = self.temp_backup_dir / f"{backup_id}.tar"
            stream = await open_stream()
            await async_add_executor_job(make_backup_dir, self.temp_backup_dir)
            f = await async_add_executor_job(path.open, 'wb')
            try:
                async for chunk in stream:
                    await async_add_executor_job(f.write, chunk)
            finally:
                await async_add_executor_job(f.close)
            remove_after_restore = True
        password_valid: bool = await self._hass.async_add_executor_job(validate_password, path, password)
        if not password_valid:
            raise IncorrectPasswordError

        def _write_restore_file() -> None:
            Path(self._hass.config.path(RESTORE_BACKUP_FILE)).write_text(
                json.dumps({
                    'path': path.as_posix(),
                    'password': password,
                    'remove_after_restore': remove_after_restore,
                    'restore_database': restore_database,
                    'restore_homeassistant': restore_homeassistant
                }),
                encoding='utf-8'
            )
        await self._hass.async_add_executor_job(_write_restore_file)
        on_progress(RestoreBackupEvent(reason=None, stage=None, state=RestoreBackupState.CORE_RESTART))
        await self._hass.services.async_call('homeassistant', 'restart', blocking=True)

    async def async_resume_restore_progress_after_restart(
        self,
        *,
        on_progress: Callable[[ManagerStateEvent], None]
    ) -> None:
        def _read_restore_file() -> Optional[Any]:
            result_path = Path(self._hass.config.path(RESTORE_BACKUP_RESULT_FILE))
            try:
                restore_result = json_util.json_loads_object(result_path.read_bytes())
            except FileNotFoundError:
                return None
            finally:
                try:
                    result_path.unlink(missing_ok=True)
                except OSError as err:
                    LOGGER.warning('Unexpected error deleting backup restore result file: %s %s', type(err), err)
            return restore_result
        restore_result = await self._hass.async_add_executor_job(_read_restore_file)
        if not restore_result:
            return
        success: bool = restore_result['success']
        if not success:
            LOGGER.warning('Backup restore failed with %s: %s', restore_result['error_type'], restore_result['error'])
        state = RestoreBackupState.COMPLETED if success else RestoreBackupState.FAILED
        on_progress(RestoreBackupEvent(reason=cast(str, restore_result['error']), stage=None, state=state))
        on_progress(IdleEvent())

def _generate_backup_id(date: str, name: str) -> str:
    return hashlib.sha1(f'{date} - {name}'.lower().encode()).hexdigest()[:8]