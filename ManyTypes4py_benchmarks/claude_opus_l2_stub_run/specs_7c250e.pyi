from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from kedro.framework.context import KedroContext
    from kedro.io import CatalogProtocol
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node


class DataCatalogSpecs:
    """Namespace that defines all specifications for a data catalog's lifecycle hooks."""

    def after_catalog_created(
        self,
        catalog: CatalogProtocol,
        conf_catalog: dict[str, Any],
        conf_creds: dict[str, Any],
        feed_dict: dict[str, Any],
        save_version: str | None,
        load_versions: dict[str, str] | None,
    ) -> None: ...


class NodeSpecs:
    """Namespace that defines all specifications for a node's lifecycle hooks."""

    def before_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> dict[str, Any] | None: ...

    def after_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None: ...

    def on_node_error(
        self,
        error: Exception,
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None: ...


class PipelineSpecs:
    """Namespace that defines all specifications for a pipeline's lifecycle hooks."""

    def before_pipeline_run(
        self,
        run_params: dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None: ...

    def after_pipeline_run(
        self,
        run_params: dict[str, Any],
        run_result: dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None: ...

    def on_pipeline_error(
        self,
        error: Exception,
        run_params: dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None: ...


class DatasetSpecs:
    """Namespace that defines all specifications for a dataset's lifecycle hooks."""

    def before_dataset_loaded(
        self,
        dataset_name: str,
        node: Node,
    ) -> None: ...

    def after_dataset_loaded(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None: ...

    def before_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None: ...

    def after_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None: ...


class KedroContextSpecs:
    """Namespace that defines all specifications for a Kedro context's lifecycle hooks."""

    def after_context_created(
        self,
        context: KedroContext,
    ) -> None: ...