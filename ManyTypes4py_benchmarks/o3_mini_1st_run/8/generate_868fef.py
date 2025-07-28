#!/usr/bin/env python3
import os
import shutil
from dataclasses import replace
from datetime import datetime
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import agate
import dbt.compilation
import dbt.exceptions
import dbt.utils
import dbt_common.utils.formatting
from dbt.adapters.events.types import BuildingCatalog, CannotGenerateDocs, CatalogWritten, WriteCatalogFailure
from dbt.adapters.factory import get_adapter
from dbt.artifacts.schemas.catalog import (
    CatalogArtifact,
    CatalogKey,
    CatalogResults,
    CatalogTable,
    ColumnMetadata,
    PrimitiveDict,
    StatsDict,
    StatsItem,
    TableMetadata,
)
from dbt.artifacts.schemas.results import NodeStatus
from dbt.constants import MANIFEST_FILE_NAME
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ResultNode
from dbt.events.types import ArtifactWritten
from dbt.exceptions import AmbiguousCatalogMatchError
from dbt.graph import ResourceTypeSelector
from dbt.graph.graph import UniqueId
from dbt.node_types import EXECUTABLE_NODE_TYPES, NodeType
from dbt.parser.manifest import write_manifest
from dbt.task.compile import CompileTask
from dbt.task.docs import DOCS_INDEX_FILE_PATH
from dbt_common.clients.system import load_file_contents
from dbt_common.dataclass_schema import ValidationError
from dbt_common.events.functions import fire_event
from dbt_common.exceptions import DbtInternalError

CATALOG_FILENAME = 'catalog.json'


def get_stripped_prefix(source: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Go through the source, extracting every key/value pair where the key starts
    with the given prefix.
    """
    cut = len(prefix)
    return {k[cut:]: v for k, v in source.items() if k.startswith(prefix)}


def build_catalog_table(data: Dict[str, Any]) -> CatalogTable:
    metadata: TableMetadata = TableMetadata.from_dict(get_stripped_prefix(data, 'table_'))
    stats: StatsDict = format_stats(get_stripped_prefix(data, 'stats:'))
    return CatalogTable(metadata=metadata, stats=stats, columns={})


class Catalog(Dict[CatalogKey, CatalogTable]):
    def __init__(self, columns: Iterable[Dict[str, Any]]) -> None:
        super().__init__()
        for col in columns:
            self.add_column(col)

    def get_table(self, data: Dict[str, Any]) -> CatalogTable:
        database = data.get('table_database')
        dkey: Optional[str] = None if database is None else str(database)
        try:
            key: CatalogKey = CatalogKey(dkey, str(data['table_schema']), str(data['table_name']))
        except KeyError as exc:
            raise dbt_common.exceptions.CompilationError(
                'Catalog information missing required key {} (got {})'.format(exc, data)
            )
        if key in self:
            table: CatalogTable = self[key]
        else:
            table = build_catalog_table(data)
            self[key] = table
        return table

    def add_column(self, data: Dict[str, Any]) -> None:
        table: CatalogTable = self.get_table(data)
        column_data: Dict[str, Any] = get_stripped_prefix(data, 'column_')
        column_data['index'] = int(column_data['index'])
        column: ColumnMetadata = ColumnMetadata.from_dict(column_data)
        table.columns[column.name] = column

    def make_unique_id_map(
        self, manifest: Manifest, selected_node_ids: Optional[Set[str]] = None
    ) -> Tuple[Dict[str, CatalogTable], Dict[str, CatalogTable]]:
        """
        Create mappings between CatalogKeys and CatalogTables for nodes and sources, filtered by selected_node_ids.

        By default, selected_node_ids is None and all nodes and sources defined in the manifest are included in the mappings.
        """
        nodes: Dict[str, CatalogTable] = {}
        sources: Dict[str, CatalogTable] = {}
        node_map, source_map = get_unique_id_mapping(manifest)
        for table in self.values():
            key = table.key()
            if key in node_map:
                unique_id: str = node_map[key]
                if selected_node_ids is None or unique_id in selected_node_ids:
                    nodes[unique_id] = replace(table, unique_id=unique_id)
            unique_ids: Set[str] = source_map.get(table.key(), set())
            for unique_id in unique_ids:
                if unique_id in sources:
                    raise AmbiguousCatalogMatchError(
                        unique_id,
                        sources[unique_id].to_dict(omit_none=True),
                        table.to_dict(omit_none=True),
                    )
                elif selected_node_ids is None or unique_id in selected_node_ids:
                    sources[unique_id] = replace(table, unique_id=unique_id)
        return (nodes, sources)


def format_stats(stats: Dict[str, Any]) -> StatsDict:
    """Given a dictionary following this layout:

        {
            'encoded:label': 'Encoded',
            'encoded:value': 'Yes',
            'encoded:description': 'Indicates if the column is encoded',
            'encoded:include': True,

            'size:label': 'Size',
            'size:value': 128,
            'size:description': 'Size of the table in MB',
            'size:include': True,
        }

    format_stats will convert the dict into a StatsDict with keys of 'encoded'
    and 'size'.
    """
    stats_collector: Dict[str, StatsItem] = {}
    base_keys: Set[str] = {k.split(':')[0] for k in stats}
    for key in base_keys:
        dct: Dict[str, Any] = {'id': key}
        for subkey in ('label', 'value', 'description', 'include'):
            dct[subkey] = stats['{}:{}'.format(key, subkey)]
        try:
            stats_item: StatsItem = StatsItem.from_dict(dct)
        except ValidationError:
            continue
        if stats_item.include:
            stats_collector[key] = stats_item
    has_stats: StatsItem = StatsItem(
        id='has_stats',
        label='Has Stats?',
        value=len(stats_collector) > 0,
        description='Indicates whether there are statistics for this table',
        include=False,
    )
    stats_collector['has_stats'] = has_stats
    return stats_collector


def mapping_key(node: Any) -> CatalogKey:
    dkey: str = dbt_common.utils.formatting.lowercase(node.database)
    return CatalogKey(dkey, node.schema.lower(), node.identifier.lower())


def get_unique_id_mapping(
    manifest: Manifest
) -> Tuple[Dict[CatalogKey, str], Dict[CatalogKey, Set[str]]]:
    node_map: Dict[CatalogKey, str] = {}
    source_map: Dict[CatalogKey, Set[str]] = {}
    for unique_id, node in manifest.nodes.items():
        key = mapping_key(node)
        node_map[key] = unique_id
    for unique_id, source in manifest.sources.items():
        key = mapping_key(source)
        if key not in source_map:
            source_map[key] = set()
        source_map[key].add(unique_id)
    return (node_map, source_map)


class GenerateTask(CompileTask):
    def run(self) -> CatalogArtifact:
        compile_results: Optional[Any] = None
        if self.args.compile:
            compile_results = CompileTask.run(self)
            if any((r.status == NodeStatus.Error for r in compile_results)):
                fire_event(CannotGenerateDocs())
                return CatalogArtifact.from_results(
                    nodes={}, sources={}, generated_at=datetime.utcnow(), errors=None, compile_results=compile_results
                )
        shutil.copyfile(DOCS_INDEX_FILE_PATH, os.path.join(self.config.project_target_path, 'index.html'))
        for asset_path in self.config.asset_paths:
            to_asset_path: str = os.path.join(self.config.project_target_path, asset_path)
            if os.path.exists(to_asset_path):
                shutil.rmtree(to_asset_path)
            if os.path.exists(asset_path):
                shutil.copytree(asset_path, to_asset_path)
        if self.manifest is None:
            raise DbtInternalError('self.manifest was None in run!')
        selected_node_ids: Optional[Set[str]] = None
        if self.args.empty_catalog:
            catalog_table: agate.Table = agate.Table([])
            exceptions: List[Exception] = []
            selected_node_ids = set()
        else:
            adapter = get_adapter(self.config)
            with adapter.connection_named('generate_catalog'):
                fire_event(BuildingCatalog())
                relations: Optional[Set[Any]] = None
                if self.job_queue is not None:
                    selected_node_ids = self.job_queue.get_selected_nodes()
                    selected_nodes: List[Any] = self._get_nodes_from_ids(self.manifest, selected_node_ids)
                    selected_source_ids: Set[str] = self._get_selected_source_ids()
                    selected_source_nodes: List[Any] = self._get_nodes_from_ids(self.manifest, selected_source_ids)
                    selected_node_ids.update(selected_source_ids)
                    selected_nodes.extend(selected_source_nodes)
                    relations = {adapter.Relation.create_from(adapter.config, node) for node in selected_nodes}  # type: ignore
                catalogable_nodes = chain(
                    [node for node in self.manifest.nodes.values() if node.is_relational and (not node.is_ephemeral_model)],
                    self.manifest.sources.values(),
                )
                used_schemas = self.manifest.get_used_schemas()
                catalog_table, exceptions = adapter.get_filtered_catalog(catalogable_nodes, used_schemas, relations)
        catalog_data: List[Dict[str, Any]] = [
            dict(zip(catalog_table.column_names, map(dbt.utils._coerce_decimal, row)))
            for row in catalog_table
        ]
        catalog: Catalog = Catalog(catalog_data)
        errors: Optional[List[str]] = None
        if exceptions:
            errors = [str(e) for e in exceptions]
        nodes, sources = catalog.make_unique_id_map(self.manifest, selected_node_ids)
        results: CatalogArtifact = self.get_catalog_results(
            nodes=nodes,
            sources=sources,
            generated_at=datetime.utcnow(),
            compile_results=compile_results,
            errors=errors,
        )
        catalog_path: str = os.path.join(self.config.project_target_path, CATALOG_FILENAME)
        results.write(catalog_path)
        fire_event(ArtifactWritten(artifact_type=results.__class__.__name__, artifact_path=catalog_path))
        if self.args.compile:
            write_manifest(self.manifest, self.config.project_target_path)
        if self.args.static:
            read_manifest_data: str = load_file_contents(os.path.join(self.config.project_target_path, MANIFEST_FILE_NAME))
            read_catalog_data: str = load_file_contents(catalog_path)
            index_data: str = load_file_contents(DOCS_INDEX_FILE_PATH)
            index_data = index_data.replace('"MANIFEST.JSON INLINE DATA"', read_manifest_data)
            index_data = index_data.replace('"CATALOG.JSON INLINE DATA"', read_catalog_data)
            static_index_path: str = os.path.join(self.config.project_target_path, 'static_index.html')
            with open(static_index_path, 'wb') as static_index_file:
                static_index_file.write(bytes(index_data, 'utf8'))
        if exceptions:
            fire_event(WriteCatalogFailure(num_exceptions=len(exceptions)))
        fire_event(CatalogWritten(path=os.path.abspath(catalog_path)))
        return results

    def get_node_selector(self) -> ResourceTypeSelector:
        if self.manifest is None or self.graph is None:
            raise DbtInternalError('manifest and graph must be set to perform node selection')
        return ResourceTypeSelector(
            graph=self.graph,
            manifest=self.manifest,
            previous_state=self.previous_state,
            resource_types=EXECUTABLE_NODE_TYPES,
            include_empty_nodes=True,
        )

    def get_catalog_results(
        self,
        nodes: Dict[str, CatalogTable],
        sources: Dict[str, CatalogTable],
        generated_at: datetime,
        compile_results: Optional[Any],
        errors: Optional[List[str]],
    ) -> CatalogArtifact:
        return CatalogArtifact.from_results(
            generated_at=generated_at, nodes=nodes, sources=sources, compile_results=compile_results, errors=errors
        )

    @classmethod
    def interpret_results(cls, results: Optional[CatalogArtifact]) -> bool:
        if results is None:
            return False
        if results.errors:
            return False
        compile_results = results._compile_results  # type: ignore
        if compile_results is None:
            return True
        return super().interpret_results(compile_results)

    @staticmethod
    def _get_nodes_from_ids(manifest: Manifest, node_ids: Iterable[str]) -> List[Any]:
        selected: List[Any] = []
        for unique_id in node_ids:
            if unique_id in manifest.nodes:
                node = manifest.nodes[unique_id]
                if node.is_relational and (not node.is_ephemeral_model):
                    selected.append(node)
            elif unique_id in manifest.sources:
                source = manifest.sources[unique_id]
                selected.append(source)
        return selected

    def _get_selected_source_ids(self) -> Set[str]:
        if self.manifest is None or self.graph is None:
            raise DbtInternalError('manifest and graph must be set to perform node selection')
        source_selector: ResourceTypeSelector = ResourceTypeSelector(
            graph=self.graph,
            manifest=self.manifest,
            previous_state=self.previous_state,
            resource_types=[NodeType.Source],
        )
        return source_selector.get_graph_queue(self.get_selection_spec()).get_selected_nodes()  # type: ignore