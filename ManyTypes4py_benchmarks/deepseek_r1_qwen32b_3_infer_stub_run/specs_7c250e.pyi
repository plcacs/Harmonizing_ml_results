"""Type stubs for specs_7c250e module."""

from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)
from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline
from kedro.framework.context import KedroContext
from kedro.io import CatalogProtocol
from pluggy import HookSpecMarker

hook_spec: HookSpecMarker = ...

class DataCatalogSpecs:
    @hook_spec
    def after_catalog_created(
        self,
        catalog: Any,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        feed_dict: Dict[str, Any],
        save_version: str,
        load_versions: Dict[str, str],
    ) -> None:
        ...

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
        ...

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
        ...

    @hook_spec
    def on_node_error(
        self,
        error: BaseException,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None:
        ...

class PipelineSpecs:
    @hook_spec
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None:
        ...

    @hook_spec
    def after_pipeline_run(
        self,
        run_params: Dict[str, Any],
        run_result: Any,
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None:
        ...

    @hook_spec
    def on_pipeline_error(
        self,
        error: BaseException,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None:
        ...

class DatasetSpecs:
    @hook_spec
    def before_dataset_loaded(
        self,
        dataset_name: str,
        node: Node,
    ) -> None:
        ...

    @hook_spec
    def after_dataset_loaded(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None:
        ...

    @hook_spec
    def before_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None:
        ...

    @hook_spec
    def after_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        node: Node,
    ) -> None:
        ...

class KedroContextSpecs:
    @hook_spec
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        ...