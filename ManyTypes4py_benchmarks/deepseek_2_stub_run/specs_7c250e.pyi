```python
from typing import Any, Optional, Dict, List, Protocol
from kedro.framework.context import KedroContext
from kedro.io import CatalogProtocol
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node

class DataCatalogSpecs:
    """Namespace that defines all specifications for a data catalog's lifecycle hooks."""
    def after_catalog_created(
        self,
        catalog: Any,
        conf_catalog: Any,
        conf_creds: Any,
        feed_dict: Any,
        save_version: Any,
        load_versions: Any
    ) -> None: ...

class NodeSpecs:
    """Namespace that defines all specifications for a node's lifecycle hooks."""
    def before_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str
    ) -> Optional[Dict[str, Any]]: ...

    def after_node_run(
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        is_async: bool,
        session_id: str
    ) -> None: ...

    def on_node_error(
        self,
        error: Exception,
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str
    ) -> None: ...

class PipelineSpecs:
    """Namespace that defines all specifications for a pipeline's lifecycle hooks."""
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol
    ) -> None: ...

    def after_pipeline_run(
        self,
        run_params: Dict[str, Any],
        run_result: Any,
        pipeline: Pipeline,
        catalog: CatalogProtocol
    ) -> None: ...

    def on_pipeline_error(
        self,
        error: Exception,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: CatalogProtocol
    ) -> None: ...

class DatasetSpecs:
    """Namespace that defines all specifications for a dataset's lifecycle hooks."""
    def before_dataset_loaded(self, dataset_name: str, node: Node) -> None: ...
    def after_dataset_loaded(self, dataset_name: str, data: Any, node: Node) -> None: ...
    def before_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None: ...
    def after_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None: ...

class KedroContextSpecs:
    """Namespace that defines all specifications for a Kedro context's lifecycle hooks."""
    def after_context_created(self, context: KedroContext) -> None: ...
```