from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING

from .markers import hook_spec

if TYPE_CHECKING:
    from kedro.framework.context import KedroContext
    from kedro.io import CatalogProtocol
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node


class DataCatalogSpecs:
    @hook_spec
    def after_catalog_created(
        self,
        catalog: CatalogProtocol,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        feed_dict: Dict[str, Any],
        save_version: Optional[Any],
        load_versions: Optional[Any],
    ) -> None:
        pass


class NodeSpecs:
    @hook_spec
    def before_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        pass

    @hook_spec
    def after_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None:
        pass

    @hook_spec
    def on_node_error(
        self,
        error: Exception,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None:
        pass


class PipelineSpecs:
    @hook_spec
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None:
        pass

    @hook_spec
    def after_pipeline_run(
        self,
        run_params: Dict[str, Any],
        run_result: Any,
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None:
        pass

    @hook_spec
    def on_pipeline_error(
        self,
        error: Exception,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None:
        pass


class DatasetSpecs:
    @hook_spec
    def before_dataset_loaded(
        self,
        dataset_name: str,
        node: Node,
    ) -> None:
        pass

    @hook_spec
    def after_dataset_loaded(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None:
        pass

    @hook_spec
    def before_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None:
        pass

    @hook_spec
    def after_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None:
        pass


class KedroContextSpecs:
    @hook_spec
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        pass