```python
from __future__ import annotations

import importlib
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, List, Optional, Union
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, track
from rich.table import Table

from prefect._experimental.sla.objects import SlaTypes
from prefect._internal.compatibility.async_dispatch import async_dispatch
from prefect._internal.concurrency.api import create_call, from_async
from prefect._internal.schemas.validators import (
    reconcile_paused_deployment,
    reconcile_schedules_runner,
)
from prefect.client.base import ServerType
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.actions import DeploymentScheduleCreate, DeploymentUpdate
from prefect.client.schemas.filters import WorkerFilter, WorkerFilterStatus
from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    ConcurrencyOptions,
)
from prefect.client.schemas.schedules import (
    SCHEDULE_TYPES,
    construct_schedule,
)
from prefect.deployments.schedules import (
    create_deployment_schedule_create,
)
from prefect.docker.docker_image import DockerImage
from prefect.events import DeploymentTriggerTypes, TriggerTypes
from prefect.exceptions import (
    ObjectNotFound,
    PrefectHTTPStatusError,
)
from prefect.runner.storage import RunnerStorage
from prefect.schedules import Schedule
from prefect.settings import (
    PREFECT_DEFAULT_WORK_POOL_NAME,
    PREFECT_UI_URL,
)
from prefect.types import ListOfNonEmptyStrings
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.asyncutils import run_coro_as_sync, sync_compatible
from prefect.utilities.callables import ParameterSchema, parameter_schema
from prefect.utilities.collections import get_from_dict, isiterable
from prefect.utilities.dockerutils import (
    parse_image_tag,
)

if TYPE_CHECKING:
    from prefect.client.types.flexible_schedule_list import FlexibleScheduleList
    from prefect.flows import Flow

__all__ = ["RunnerDeployment"]


class DeploymentApplyError(RuntimeError):
    """
    Raised when an error occurs while applying a deployment.
    """


class RunnerDeployment(BaseModel):
    """
    A Prefect RunnerDeployment definition, used for specifying and building deployments.

    Attributes:
        name: A name for the deployment (required).
        version: An optional version for the deployment; defaults to the flow's version
        description: An optional description of the deployment; defaults to the flow's
            description
        tags: An optional list of tags to associate with this deployment; note that tags
            are used only for organizational purposes. For delegating work to workers,
            see `work_queue_name`.
        schedule: A schedule to run this deployment on, once registered
        parameters: A dictionary of parameter values to pass to runs created from this
            deployment
        path: The path to the working directory for the workflow, relative to remote
            storage or, if stored on a local filesystem, an absolute path
        entrypoint: The path to the entrypoint for the workflow, always relative to the
            `path`
        parameter_openapi_schema: The parameter schema of the flow, including defaults.
        enforce_parameter_schema: Whether or not the Prefect API should enforce the
            parameter schema for this deployment.
        work_pool_name: The name of the work pool to use for this deployment.
        work_queue_name: The name of the work queue to use for this deployment's scheduled runs.
            If not provided the default work queue for the work pool will be used.
        job_variables: Settings used to override the values specified default base job template
            of the chosen work pool. Refer to the base job template of the chosen work pool for
            available settings.
        _sla: (Experimental) SLA configuration for the deployment. May be removed or modified at any time. Currently only supported on Prefect Cloud.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="The name of the deployment.")
    flow_name: Optional[str] = Field(
        None, description="The name of the underlying flow; typically inferred."
    )
    description: Optional[str] = Field(
        default=None, description="An optional description of the deployment."
    )
    version: Optional[str] = Field(
        default=None, description="An optional version for the deployment."
    )
    tags: ListOfNonEmptyStrings = Field(
        default_factory=list,
        description="One of more tags to apply to this deployment.",
    )
    schedules: Optional[List[DeploymentScheduleCreate]] = Field(
        default=None,
        description="The schedules that should cause this deployment to run.",
    )
    concurrency_limit: Optional[int] = Field(
        default=None,
        description="The maximum number of concurrent runs of this deployment.",
    )
    concurrency_options: Optional[ConcurrencyOptions] = Field(
        default=None,
        description="The concurrency limit config for the deployment.",
    )
    paused: Optional[bool] = Field(
        default=None, description="Whether or not the deployment is paused."
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    entrypoint: Optional[str] = Field(
        default=None,
        description=(
            "The path to the entrypoint for the workflow, relative to the `path`."
        ),
    )
    triggers: List[Union[DeploymentTriggerTypes, TriggerTypes]] = Field(
        default_factory=list,
        description="The triggers that should cause this deployment to run.",
    )
    enforce_parameter_schema: bool] = Field(
        default=True,
        description=(
            "Whether or not the Prefect API should enforce the parameter schema for"
            " this deployment."
        ),
    )
    storage: Optional[RunnerStorage] = Field(
        default=None,
        description=(
            "The storage object used to retrieve flow code for this deployment."
        ),
    )
    work_pool_name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the work pool to use for this deployment. Only used when"
            " the deployment is registered with a built runner."
        ),
    )
    work_queue_name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the work queue to use for this deployment. Only used when"
            " the deployment is registered with a built runner."
        ),
    )
    job_variables: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Job variables used to override the default values of a work pool"
            " base job template. Only used when the deployment is registered with"
            " a built runner."
        ),
    )
    # (Experimental) SLA configuration for the deployment. May be removed or modified at any time. Currently only supported on Prefect Cloud.
    _sla: Optional[Union[SlaTypes, list[SlaTypes]]] = PrivateAttr(
        default=None,
    )
    _entrypoint_type: EntrypointType = PrivateAttr(
        default=EntrypointType.FILE_PATH,
    )
    _path: Optional[str] = PrivateAttr(
        default=None,
    )
    _parameter_openapi_schema: ParameterSchema = PrivateAttr(
        default_factory=ParameterSchema,
    )

    @property
    def entrypoint_type(self) -> EntrypointType:
        return self._entrypoint_type

    @property
    def full_name(self) -> str:
        return f"{self.flow_name}/{self.name}"

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value.endswith(".py"):
            return Path(value).stem
        return value

    @model_validator(mode="after")
    def validate_automation_names(self) -> RunnerDeployment:
        """Ensure that each trigger has a name for its automation if none is provided."""
        trigger: Union[DeploymentTriggerTypes, TriggerTypes]
        for i, trigger in enumerate(self.triggers, start=1):
            if trigger.name is None:
                trigger.name = f"{self.name}__automation_{i}"
        return self

    @model_validator(mode="before")
    @classmethod
    def reconcile_paused(cls, values: dict) -> dict:
        return reconcile_paused_deployment(values)

    @model_validator(mode="before")
    @classmethod
    def reconcile_schedules(cls, values: dict) -> dict:
        return reconcile_schedules_runner(values)

    async def _create(
        self, work_pool_name: Optional[str] = None, image: Optional[str] = None
    ) -> UUID:
        work_pool_name = work_pool_name or self.work_pool_name

        if image and not work_pool_name:
            raise ValueError(
                "An image can only be provided when registering a deployment with a"
                " work pool."
            )

        if self.work_queue_name and not work_pool_name:
            raise ValueError(
                "A work queue can only be provided when registering a deployment with"
                " a work pool."
            )

        if self.job_variables and not work_pool_name:
            raise ValueError(
                "Job variables can only be provided when registering a deployment"
                " with a work pool."
            )

        async with get_client() as client:
            flow_id = await client.create_flow_from_name(self.flow_name)

            create_payload: dict[str, Any] = dict(
                flow_id=flow_id,
                name=self.name,
                work_queue_name=self.work_queue_name,
                work_pool_name=work_pool_name,
                version=self.version,
                paused=self.paused,
                schedules=self.schedules,
                concurrency_limit=self.concurrency_limit,
                concurrency_options=self.concurrency_options,
                parameters=self.parameters,
                description=self.description,
                tags=self.tags,
                path=self._path,
                entrypoint=self.entrypoint,
                storage_document_id=None,
                infrastructure_document_id=None,
                parameter_openapi_schema=self._parameter_openapi_schema.model_dump(
                    exclude_unset=True
                ),
                enforce_parameter_schema=self.enforce_parameter_schema,
            )

            if work_pool_name:
                create_payload["job_variables"] = self.job_variables
                if image:
                    create_payload["job_variables"]["image"] = image
                create_payload["path"] = None if self.storage else self._path
                if self.storage:
                    pull_steps = self.storage.to_pull_step()
                    if isinstance(pull_steps, list):
                        create_payload["pull_steps"] = pull_steps
                    else:
                        create_payload["pull_steps"] = [pull_steps]
                else:
                    create_payload["pull_steps"] = []

            try:
                deployment_id = await client.create_deployment(**create_payload)
            except Exception as exc:
                if isinstance(exc, PrefectHTTPStatusError):
                    detail = exc.response.json().get("detail")
                    if detail:
                        raise DeploymentApplyError(detail) from exc
                raise DeploymentApplyError(
                    f"Error while applying deployment: {str(exc)}"
                ) from exc

            await self._create_triggers(deployment_id, client)

            # We plan to support SLA configuration on the Prefect Server in the future.
            # For now, we only support it on Prefect Cloud.

            # If we're provided with an empty list, we will call the apply endpoint
            # to remove existing SLAs for the deployment. If the argument is not provided,
            # we will not call the endpoint.
            if self._sla or self._sla == []:
                await self._create_slas(deployment_id, client)

            return deployment_id

    async def _update(self, deployment_id: UUID, client: PrefectClient) -> UUID:
        parameter_openapi_schema = self._parameter_openapi_schema.model_dump(
            exclude_unset=True
        )
        await client.update_deployment(
            deployment_id,
            deployment=DeploymentUpdate(
                parameter_openapi_schema=parameter_openapi_schema,
                **self.model_dump(
                    mode="json",
                    exclude_unset=True,
                    exclude={"storage", "name", "flow_name", "triggers"},
                ),
            ),
        )

        await self._create_triggers(deployment_id, client)

        # We plan to support SLA configuration on the Prefect Server in the future.
        # For now, we only support it on Prefect Cloud.

        # If we're provided with an empty list, we will call the apply endpoint
        # to remove existing SLAs for the deployment. If the argument is not provided,
        # we will not call the endpoint.
        if self._sla or self._sla == []:
            await self._create_slas(deployment_id, client)

        return deployment_id

    async def _create_triggers(self, deployment_id: UUID, client: PrefectClient) -> None:
        try:
            # The triggers defined in the deployment spec are, essentially,
            # anonymous and attempting truly sync them with cloud is not
            # feasible. Instead, we remove all automations that are owned
            # by the deployment, meaning that they were created via this
            # mechanism below, and then recreate them.
            await client.delete_resource_owned_automations(
                f"prefect.deployment.{deployment_id}"
            )
        except PrefectHTTPStatusError as e:
            if e.response.status_code == 404:
                # This Prefect server does not support automations, so we can safely
                # ignore this 404 and move on.
                return
            raise e

        for trigger in self.triggers:
            trigger.set_deployment_id(deployment_id)
            await client.create_automation(trigger.as_automation())

    @sync_compatible
    async def apply(
        self, work_pool_name: Optional[str] = None, image: Optional[str] = None
    ) -> UUID:
        """
        Registers this deployment with the API and returns the deployment's ID.

        Args:
            work_pool_name: The name of the work pool to use for this
                deployment.
            image: The registry, name, and tag of the Docker image to
                use for this deployment. Only used when the deployment is
                deployed to a work pool.

        Returns:
            The ID of the created deployment.
        """

        async with get_client() as client:
            try:
                deployment = await client.read_deployment_by_name(self.full_name)
            except ObjectNotFound:
                return await self._create(work_pool_name, image)
            else:
                if image:
                    self.job_variables["image"] = image
                if work_pool_name:
                    self.work_pool_name = work_pool_name
                return await self._update(deployment.id, client)

    async def _create_slas(self, deployment_id: UUID, client: PrefectClient) -> None:
        if not isinstance(self._sla, list):
            self._sla = [self._sla]

        if client.server_type == ServerType.CLOUD:
            await client.apply_slas_for_deployment(deployment_id, self._sla)
        else:
            raise ValueError(
                "SLA configuration is currently only supported on Prefect Cloud."
            )

    @staticmethod
    def _construct_deployment_schedules(
        interval: Optional[
            Union[Iterable[Union[int, float, timedelta]], int, float, timedelta]
        ] = None,
        anchor_date: Optional[Union[datetime, str]] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        timezone: Optional[str] = None,
        schedule: Union[SCHEDULE_TYPES, Schedule, None] = None,
        schedules: Optional["FlexibleScheduleList"] = None,
    ) -> Union[List[DeploymentScheduleCreate], "FlexibleScheduleList"]:
        """
        Construct a schedule or schedules from the provided arguments.

        This method serves as a unified interface for creating deployment
        schedules. If `schedules` is provided, it is directly returned. If
        `schedule` is provided, it is encapsulated in a list and returned. If
        `interval`, `cron`, or `rrule` are provided, they are used to construct
        schedule objects.

        Args:
            interval: An interval on which to schedule runs, either as a single
              value or as a list of values. Accepts numbers (interpreted as
              seconds) or `timedelta` objects. Each value defines a separate
              scheduling interval.
            anchor_date: The anchor date from which interval schedules should
              start. This applies to all intervals if a list is provided.
            cron: A cron expression or a list of cron expressions defining cron
              schedules. Each expression defines a separate cron schedule.
            rrule: An rrule string or a list of rrule strings for scheduling.
              Each string defines a separate recurrence rule.
            timezone: The timezone to apply to the cron or rrule schedules.
              This is a single value applied uniformly to all schedules.
            schedule: A singular schedule object, used for advanced scheduling
              options like specifying a timezone. This is returned as a list
              containing this single schedule.
            schedules: A pre-defined list of schedule objects. If provided,
              this list is returned as-is, bypassing other schedule construction
              logic.
        """
        num_schedules = sum(
            1
            for entry in (interval, cron, rrule, schedule, schedules)
            if entry is not None
        )
        if num_schedules > 1:
            raise ValueError(
                "Only one of interval, cron, rrule, schedule, or schedules can be provided."
            )
        elif num_schedules == 0:
            return []

        if schedules is not None:
            return schedules
        elif interval or cron or rrule:
            # `interval`, `cron`, and `rrule` can be lists of values. This
            # block figures out which one is not None and uses that to
            # construct the list of schedules via `construct_schedule`.
            parameters = [("interval", interval), ("cron", cron), ("rrule", rrule)]
            schedule_type, value = [
                param for param in parameters if param[1] is not None
            ][0]

            if not