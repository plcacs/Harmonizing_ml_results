from __future__ import annotations
import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import agate
from dbt.artifacts.resources import CompiledResource
from dbt.artifacts.schemas.base import (
    ArtifactMixin,
    BaseArtifactMetadata,
    get_artifact_schema_version,
    schema_version,
)
from dbt.artifacts.schemas.batch_results import BatchResults
from dbt.artifacts.schemas.results import BaseResult, ExecutionResult, NodeResult, ResultNode, RunStatus
from dbt.exceptions import scrub_secrets
from dbt_common.clients.system import write_json
from dbt_common.constants import SECRET_ENV_PREFIX


@dataclass
class RunResult(NodeResult):
    agate_table: Optional[agate.Table] = field(
        default=None,
        metadata={'serialize': lambda x: None, 'deserialize': lambda x: None}
    )
    batch_results: Optional[Any] = None

    @property
    def skipped(self) -> bool:
        return self.status == RunStatus.Skipped

    @classmethod
    def from_node(cls, node: Any, status: RunStatus, message: str) -> RunResult:
        thread_id: str = threading.current_thread().name
        return RunResult(
            status=status,
            thread_id=thread_id,
            execution_time=0,
            timing=[],
            message=message,
            node=node,
            adapter_response={},
            failures=None,
            batch_results=None
        )


@dataclass
class RunResultsMetadata(BaseArtifactMetadata):
    dbt_schema_version: str = field(default_factory=lambda: str(RunResultsArtifact.dbt_schema_version))


@dataclass
class RunResultOutput(BaseResult):
    batch_results: Optional[Any] = None


def process_run_result(result: RunResult) -> RunResultOutput:
    compiled: bool = isinstance(result.node, CompiledResource)
    return RunResultOutput(
        unique_id=result.node.unique_id,
        status=result.status,
        timing=result.timing,
        thread_id=result.thread_id,
        execution_time=result.execution_time,
        message=result.message,
        adapter_response=result.adapter_response,
        failures=result.failures,
        batch_results=result.batch_results,
        compiled=result.node.compiled if compiled else None,
        compiled_code=result.node.compiled_code if compiled else None,
        relation_name=result.node.relation_name if compiled else None
    )


@dataclass
class RunExecutionResult(ExecutionResult):
    args: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def write(self, path: str) -> None:
        writable: RunResultsArtifact = RunResultsArtifact.from_execution_results(
            results=self.results,
            elapsed_time=self.elapsed_time,
            generated_at=self.generated_at,
            args=self.args
        )
        writable.write(path)


@dataclass
@schema_version('run-results', 6)
class RunResultsArtifact(ExecutionResult, ArtifactMixin):
    args: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_execution_results(
        cls,
        results: Iterable[Any],
        elapsed_time: float,
        generated_at: datetime,
        args: Dict[str, Any]
    ) -> RunResultsArtifact:
        processed_results: List[RunResultOutput] = [
            process_run_result(result) for result in results if isinstance(result, RunResult)
        ]
        meta: RunResultsMetadata = RunResultsMetadata(
            dbt_schema_version=str(cls.dbt_schema_version),
            generated_at=generated_at  # assuming metadata can accept an extra key
        )
        secret_vars: List[str] = [
            v for k, v in args['vars'].items() if k.startswith(SECRET_ENV_PREFIX) and v.strip()
        ]
        scrubbed_args: Dict[str, Any] = copy.deepcopy(args)
        scrubbed_args['invocation_command'] = scrub_secrets(scrubbed_args['invocation_command'], secret_vars)
        scrubbed_args['vars'] = {k: scrub_secrets(v, secret_vars) for k, v in scrubbed_args['vars'].items()}
        return cls(
            metadata=meta,
            results=processed_results,
            elapsed_time=elapsed_time,
            args=scrubbed_args
        )

    @classmethod
    def compatible_previous_versions(cls) -> List[Tuple[str, int]]:
        return [('run-results', 4), ('run-results', 5)]

    @classmethod
    def upgrade_schema_version(cls, data: Dict[str, Any]) -> RunResultsArtifact:
        """This overrides the "upgrade_schema_version" call in VersionedSchema (via
        ArtifactMixin) to modify the dictionary passed in from earlier versions of the run_results.
        """
        run_results_schema_version: int = get_artifact_schema_version(data)
        if run_results_schema_version <= 5:
            for result in data['results']:
                result['compiled'] = False
                result['compiled_code'] = ''
                result['relation_name'] = ''
        return cls.from_dict(data)

    def write(self, path: str) -> None:
        write_json(path, self.to_dict(omit_none=False))