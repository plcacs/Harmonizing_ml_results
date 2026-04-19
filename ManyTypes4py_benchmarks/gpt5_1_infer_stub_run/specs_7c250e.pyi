from typing import Any, Dict, Optional
from kedro.framework.context import KedroContext
from kedro.io import CatalogProtocol
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node

class DataCatalogSpecs:
    def after_catalog_created(
        self,
        catalog: CatalogProtocol,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        feed_dict: Dict[str, Any],
        save_version: Optional[str],
        load_versions: Optional[Dict[str, str]],
    ) -> None: ...

class NodeSpecs:
    def before_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> Optional[Dict[str, Any]]: ...
    def after_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None: ...
    def on_node_error(
        self,
        error: Exception,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> None: ...

class PipelineSpecs:
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None: ...
    def after_pipeline_run(
        self,
        run_params: Dict[str, Any],
        run_result: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None: ...
    def on_pipeline_error(
        self,
        error: Exception,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol,
    ) -> None: ...

class DatasetSpecs:
    def before_dataset_loaded(self, dataset_name: str, node: Node) -> None: ...
    def after_dataset_loaded(self, dataset_name: str, data: Any, node: Node) -> None: ...
    def before_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None: ...
    def after_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None: ...

class KedroContextSpecs:
    def after_context_created(self, context: KedroContext) -> None: ...