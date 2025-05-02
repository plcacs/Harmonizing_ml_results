from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from .markers import hook_spec

if TYPE_CHECKING:
    from kedro.framework.context import KedroContext
    from kedro.io import CatalogProtocol
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node

class DataCatalogSpecs:
    """Namespace that defines all specifications for a data catalog's lifecycle hooks."""

    @hook_spec
    def after_catalog_created(
        self,
        catalog: CatalogProtocol,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        feed_dict: Dict[str, Any],
        save_version: str,
        load_versions: Dict[str, str]
    ) -> None:
        pass

class NodeSpecs:
    """Namespace that defines all specifications for a node's lifecycle hooks."""

    @hook_spec
    def before_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str
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
        session_id: str
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
        session_id: str
    ) -> None:
        pass

class PipelineSpecs:
    """Namespace that defines all specifications for a pipeline's lifecycle hooks."""

    @hook_spec
    def before_pipeline_run(
        self,
        run_params: Dict[str, Union[str, List[str], Dict[str, Any], None]],
        pipeline: Pipeline,
        catalog: CatalogProtocol
    ) -> None:
        pass

    @hook_spec
    def after_pipeline_run(
        self,
        run_params: Dict[str, Union[str, List[str], Dict[str, Any], None]],
        run_result: Any,
        pipeline: Pipeline,
        catalog: CatalogProtocol
    ) -> None:
        pass

    @hook_spec
    def on_pipeline_error(
        self,
        error: Exception,
        run_params: Dict[str, Union[str, List[str], Dict[str, Any], None]],
        pipeline: Pipeline,
        catalog: CatalogProtocol
    ) -> None:
        pass

class DatasetSpecs:
    """Namespace that defines all specifications for a dataset's lifecycle hooks."""

    @hook_spec
    def before_dataset_loaded(
        self,
        dataset_name: str,
        node: Node
    ) -> None:
        pass

    @hook_spec
    def after_dataset_loaded(
        self,
        dataset_name: str,
        data: Any,
        node: Node
    ) -> None:
        pass

    @hook_spec
    def before_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        node: Node
    ) -> None:
        pass

    @hook_spec
    def after_dataset_saved(
        self,
        dataset_name: str,
        data: Any,
        node: Node
    ) -> None:
        pass

class KedroContextSpecs:
    """Namespace that defines all specifications for a Kedro context's lifecycle hooks."""

    @hook_spec
    def after_context_created(
        self,
        context: KedroContext
    ) -> None:
        pass
