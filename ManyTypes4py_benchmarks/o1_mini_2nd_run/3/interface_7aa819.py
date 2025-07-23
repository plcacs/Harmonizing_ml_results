from collections.abc import Hashable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Tuple
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

_UniqueKey: TypeAlias = Tuple[Hashable, ...]

class DBSingleton(type):
    """Ensures that only one database interface is created per unique key"""
    _instances: Dict[_UniqueKey, 'PrefectDBInterface'] = {}

    def __call__(
        cls,
        *args: Any,
        database_config: BaseDatabaseConfiguration,
        query_components: 'BaseQueryComponents',
        orm: Any,
        **kwargs: Any
    ) -> 'PrefectDBInterface':
        instance_key: _UniqueKey = (
            cls.__name__,
            database_config.unique_key(),
            query_components.unique_key(),
            orm.unique_key(),
        )
        try:
            instance: 'PrefectDBInterface' = cls._instances[instance_key]
        except KeyError:
            instance = cls._instances[instance_key] = super().__call__(
                *args,
                database_config=database_config,
                query_components=query_components,
                orm=orm,
                **kwargs
            )
        return instance

class PrefectDBInterface(metaclass=DBSingleton):
    """
    An interface for backend-specific SqlAlchemy actions and ORM models.

    The REST API can be configured to run against different databases in order maintain
    performance at different scales. This interface integrates database- and dialect-
    specific configuration into a unified interface that the orchestration engine runs
    against.
    """

    def __init__(
        self,
        database_config: BaseDatabaseConfiguration,
        query_components: 'BaseQueryComponents',
        orm: Any
    ) -> None:
        self.database_config: BaseDatabaseConfiguration = database_config
        self.queries: 'BaseQueryComponents' = query_components
        self.orm: Any = orm

    async def create_db(self) -> None:
        """Create the database"""
        await self.run_migrations_upgrade()

    async def drop_db(self) -> None:
        """Drop the database"""
        await self.run_migrations_downgrade(revision='base')

    async def run_migrations_upgrade(self) -> None:
        """Run all upgrade migrations"""
        await run_sync_in_worker_thread(alembic_upgrade)

    async def run_migrations_downgrade(self, revision: str = '-1') -> None:
        """Run all downgrade migrations"""
        await run_sync_in_worker_thread(alembic_downgrade, revision=revision)

    async def is_db_connectable(self) -> bool:
        """
        Returns boolean indicating if the database is connectable.
        This method is used to determine if the server is ready to accept requests.
        """
        engine: AsyncEngine = await self.engine()
        try:
            async with engine.connect():
                return True
        except Exception:
            return False

    async def engine(self) -> AsyncEngine:
        """
        Provides a SqlAlchemy engine against a specific database.
        """
        engine: AsyncEngine = await self.database_config.engine()
        return engine

    async def session(self) -> AsyncSession:
        """
        Provides a SQLAlchemy session.
        """
        engine: AsyncEngine = await self.engine()
        session: AsyncSession = await self.database_config.session(engine)
        return session

    @asynccontextmanager
    async def session_context(
        self,
        begin_transaction: bool = False,
        with_for_update: bool = False
    ) -> AsyncIterator[AsyncSession]:
        """
        Provides a SQLAlchemy session and a context manager for opening/closing
        the underlying connection.

        Args:
            begin_transaction: if True, the context manager will begin a SQL transaction.
                Exiting the context manager will COMMIT or ROLLBACK any changes.
        """
        session: AsyncSession = await self.session()
        async with session:
            if begin_transaction:
                async with self.database_config.begin_transaction(
                    session, with_for_update=with_for_update
                ):
                    yield session
            else:
                yield session

    @property
    def dialect(self) -> str:
        return get_dialect(self.database_config.connection_url)

    @property
    def Base(self) -> Any:
        """Base class for orm models"""
        return orm_models.Base

    @property
    def Flow(self) -> Any:
        """A flow orm model"""
        return orm_models.Flow

    @property
    def FlowRun(self) -> Any:
        """A flow run orm model"""
        return orm_models.FlowRun

    @property
    def FlowRunState(self) -> Any:
        """A flow run state orm model"""
        return orm_models.FlowRunState

    @property
    def TaskRun(self) -> Any:
        """A task run orm model"""
        return orm_models.TaskRun

    @property
    def TaskRunState(self) -> Any:
        """A task run state orm model"""
        return orm_models.TaskRunState

    @property
    def Artifact(self) -> Any:
        """An artifact orm model"""
        return orm_models.Artifact

    @property
    def ArtifactCollection(self) -> Any:
        """An artifact collection orm model"""
        return orm_models.ArtifactCollection

    @property
    def TaskRunStateCache(self) -> Any:
        """A task run state cache orm model"""
        return orm_models.TaskRunStateCache

    @property
    def Deployment(self) -> Any:
        """A deployment orm model"""
        return orm_models.Deployment

    @property
    def DeploymentSchedule(self) -> Any:
        """A deployment schedule orm model"""
        return orm_models.DeploymentSchedule

    @property
    def SavedSearch(self) -> Any:
        """A saved search orm model"""
        return orm_models.SavedSearch

    @property
    def WorkPool(self) -> Any:
        """A work pool orm model"""
        return orm_models.WorkPool

    @property
    def Worker(self) -> Any:
        """A worker process orm model"""
        return orm_models.Worker

    @property
    def Log(self) -> Any:
        """A log orm model"""
        return orm_models.Log

    @property
    def ConcurrencyLimit(self) -> Any:
        """A concurrency model"""
        return orm_models.ConcurrencyLimit

    @property
    def ConcurrencyLimitV2(self) -> Any:
        """A v2 concurrency model"""
        return orm_models.ConcurrencyLimitV2

    @property
    def CsrfToken(self) -> Any:
        """A csrf token model"""
        return orm_models.CsrfToken

    @property
    def WorkQueue(self) -> Any:
        """A work queue model"""
        return orm_models.WorkQueue

    @property
    def Agent(self) -> Any:
        """An agent model"""
        return orm_models.Agent

    @property
    def BlockType(self) -> Any:
        """A block type model"""
        return orm_models.BlockType

    @property
    def BlockSchema(self) -> Any:
        """A block schema model"""
        return orm_models.BlockSchema

    @property
    def BlockSchemaReference(self) -> Any:
        """A block schema reference model"""
        return orm_models.BlockSchemaReference

    @property
    def BlockDocument(self) -> Any:
        """A block document model"""
        return orm_models.BlockDocument

    @property
    def BlockDocumentReference(self) -> Any:
        """A block document reference model"""
        return orm_models.BlockDocumentReference

    @property
    def FlowRunNotificationPolicy(self) -> Any:
        """A flow run notification policy model"""
        return orm_models.FlowRunNotificationPolicy

    @property
    def FlowRunNotificationQueue(self) -> Any:
        """A flow run notification queue model"""
        return orm_models.FlowRunNotificationQueue

    @property
    def Configuration(self) -> Any:
        """An configuration model"""
        return orm_models.Configuration

    @property
    def Variable(self) -> Any:
        """A variable model"""
        return orm_models.Variable

    @property
    def FlowRunInput(self) -> Any:
        """A flow run input model"""
        return orm_models.FlowRunInput

    @property
    def Automation(self) -> Any:
        """An automation model"""
        return orm_models.Automation

    @property
    def AutomationBucket(self) -> Any:
        """An automation bucket model"""
        return orm_models.AutomationBucket

    @property
    def AutomationRelatedResource(self) -> Any:
        """An automation related resource model"""
        return orm_models.AutomationRelatedResource

    @property
    def CompositeTriggerChildFiring(self) -> Any:
        """A model capturing a composite trigger's child firing"""
        return orm_models.CompositeTriggerChildFiring

    @property
    def AutomationEventFollower(self) -> Any:
        """A model capturing one event following another event"""
        return orm_models.AutomationEventFollower

    @property
    def Event(self) -> Any:
        """An event model"""
        return orm_models.Event

    @property
    def EventResource(self) -> Any:
        """An event resource model"""
        return orm_models.EventResource
