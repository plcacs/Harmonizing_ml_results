from collections.abc import Hashable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from typing_extensions import TypeAlias
from prefect.server.database import orm_models
from prefect.server.database.alembic_commands import alembic_downgrade, alembic_upgrade
from prefect.server.database.configurations import BaseDatabaseConfiguration
from prefect.server.utilities.database import get_dialect
from prefect.utilities.asyncutils import run_sync_in_worker_thread
if TYPE_CHECKING:
    from prefect.server.database.query_components import BaseQueryComponents
_UniqueKey = tuple[Hashable, ...]

class DBSingleton(type):
    _instances: dict = dict()

    def __call__(cls, *args, database_config: BaseDatabaseConfiguration, query_components: BaseQueryComponents, orm: Any, **kwargs) -> Any:
        instance_key = (cls.__name__, database_config.unique_key(), query_components.unique_key(), orm.unique_key())
        try:
            instance = cls._instances[instance_key]
        except KeyError:
            instance = cls._instances[instance_key] = super().__call__(*args, database_config=database_config, query_components=query_components, orm=orm, **kwargs)
        return instance

class PrefectDBInterface(metaclass=DBSingleton):
    def __init__(self, database_config: BaseDatabaseConfiguration, query_components: BaseQueryComponents, orm: Any) -> None:
        self.database_config = database_config
        self.queries = query_components
        self.orm = orm

    async def create_db(self) -> None:
        await self.run_migrations_upgrade()

    async def drop_db(self) -> None:
        await self.run_migrations_downgrade(revision='base')

    async def run_migrations_upgrade(self) -> None:
        await run_sync_in_worker_thread(alembic_upgrade)

    async def run_migrations_downgrade(self, revision: str = '-1') -> None:
        await run_sync_in_worker_thread(alembic_downgrade, revision=revision)

    async def is_db_connectable(self) -> bool:
        engine = await self.engine()
        try:
            async with engine.connect():
                return True
        except Exception:
            return False

    async def engine(self) -> AsyncEngine:
        engine = await self.database_config.engine()
        return engine

    async def session(self) -> AsyncSession:
        engine = await self.engine()
        return await self.database_config.session(engine)

    @asynccontextmanager
    async def session_context(self, begin_transaction: bool = False, with_for_update: bool = False):
        session = await self.session()
        async with session:
            if begin_transaction:
                async with self.database_config.begin_transaction(session, with_for_update=with_for_update):
                    yield session
            else:
                yield session

    @property
    def dialect(self) -> str:
        return get_dialect(self.database_config.connection_url)

    @property
    def Base(self) -> Any:
        return orm_models.Base

    @property
    def Flow(self) -> Any:
        return orm_models.Flow

    @property
    def FlowRun(self) -> Any:
        return orm_models.FlowRun

    @property
    def FlowRunState(self) -> Any:
        return orm_models.FlowRunState

    @property
    def TaskRun(self) -> Any:
        return orm_models.TaskRun

    @property
    def TaskRunState(self) -> Any:
        return orm_models.TaskRunState

    @property
    def Artifact(self) -> Any:
        return orm_models.Artifact

    @property
    def ArtifactCollection(self) -> Any:
        return orm_models.ArtifactCollection

    @property
    def TaskRunStateCache(self) -> Any:
        return orm_models.TaskRunStateCache

    @property
    def Deployment(self) -> Any:
        return orm_models.Deployment

    @property
    def DeploymentSchedule(self) -> Any:
        return orm_models.DeploymentSchedule

    @property
    def SavedSearch(self) -> Any:
        return orm_models.SavedSearch

    @property
    def WorkPool(self) -> Any:
        return orm_models.WorkPool

    @property
    def Worker(self) -> Any:
        return orm_models.Worker

    @property
    def Log(self) -> Any:
        return orm_models.Log

    @property
    def ConcurrencyLimit(self) -> Any:
        return orm_models.ConcurrencyLimit

    @property
    def ConcurrencyLimitV2(self) -> Any:
        return orm_models.ConcurrencyLimitV2

    @property
    def CsrfToken(self) -> Any:
        return orm_models.CsrfToken

    @property
    def WorkQueue(self) -> Any:
        return orm_models.WorkQueue

    @property
    def Agent(self) -> Any:
        return orm_models.Agent

    @property
    def BlockType(self) -> Any:
        return orm_models.BlockType

    @property
    def BlockSchema(self) -> Any:
        return orm_models.BlockSchema

    @property
    def BlockSchemaReference(self) -> Any:
        return orm_models.BlockSchemaReference

    @property
    def BlockDocument(self) -> Any:
        return orm_models.BlockDocument

    @property
    def BlockDocumentReference(self) -> Any:
        return orm_models.BlockDocumentReference

    @property
    def FlowRunNotificationPolicy(self) -> Any:
        return orm_models.FlowRunNotificationPolicy

    @property
    def FlowRunNotificationQueue(self) -> Any:
        return orm_models.FlowRunNotificationQueue

    @property
    def Configuration(self) -> Any:
        return orm_models.Configuration

    @property
    def Variable(self) -> Any:
        return orm_models.Variable

    @property
    def FlowRunInput(self) -> Any:
        return orm_models.FlowRunInput

    @property
    def Automation(self) -> Any:
        return orm_models.Automation

    @property
    def AutomationBucket(self) -> Any:
        return orm_models.AutomationBucket

    @property
    def AutomationRelatedResource(self) -> Any:
        return orm_models.AutomationRelatedResource

    @property
    def CompositeTriggerChildFiring(self) -> Any:
        return orm_models.CompositeTriggerChildFiring

    @property
    def AutomationEventFollower(self) -> Any:
        return orm_models.AutomationEventFollower

    @property
    def Event(self) -> Any:
        return orm_models.Event

    @property
    def EventResource(self) -> Any:
        return orm_models.EventResource
