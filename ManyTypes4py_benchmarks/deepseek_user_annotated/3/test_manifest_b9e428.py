import os
import unittest
from argparse import Namespace
from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from itertools import product
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from unittest import mock

import freezegun
import pytest

import dbt.flags
import dbt.version
import dbt_common.invocation
from dbt import tracking
from dbt.adapters.base.plugin import AdapterPlugin
from dbt.artifacts.resources import (
    ExposureType,
    MaturityType,
    MetricInputMeasure,
    MetricTypeParams,
    Owner,
    RefArgs,
    WhereFilter,
    WhereFilterIntersection,
)
from dbt.contracts.files import FileHash
from dbt.contracts.graph.manifest import DisabledLookup, Manifest, ManifestMetadata
from dbt.contracts.graph.nodes import (
    DependsOn,
    Exposure,
    Group,
    Metric,
    ModelConfig,
    ModelNode,
    SeedNode,
    SourceDefinition,
)
from dbt.exceptions import AmbiguousResourceNameRefError, ParsingError
from dbt.flags import set_from_args
from dbt.node_types import NodeType
from dbt_common.events.functions import reset_metadata_vars
from dbt_semantic_interfaces.type_enums import MetricType
from tests.unit.utils import (
    MockDocumentation,
    MockGenerateMacro,
    MockMacro,
    MockMaterialization,
    MockNode,
    MockSource,
    inject_plugin,
    make_manifest,
)

REQUIRED_PARSED_NODE_KEYS: FrozenSet[str] = frozenset(
    {
        "alias",
        "tags",
        "config",
        "unique_id",
        "refs",
        "sources",
        "metrics",
        "meta",
        "depends_on",
        "database",
        "schema",
        "name",
        "resource_type",
        "group",
        "package_name",
        "path",
        "original_file_path",
        "raw_code",
        "language",
        "description",
        "primary_key",
        "columns",
        "fqn",
        "build_path",
        "compiled_path",
        "patch_path",
        "docs",
        "doc_blocks",
        "checksum",
        "unrendered_config",
        "unrendered_config_call_dict",
        "created_at",
        "config_call_dict",
        "relation_name",
        "contract",
        "access",
        "version",
        "latest_version",
        "constraints",
        "deprecation_date",
        "defer_relation",
        "time_spine",
        "batch",
        "freshness",
    }
)

REQUIRED_COMPILED_NODE_KEYS: FrozenSet[str] = frozenset(
    REQUIRED_PARSED_NODE_KEYS
    | {"compiled", "extra_ctes_injected", "extra_ctes", "compiled_code", "relation_name"}
)

ENV_KEY_NAME: str = "KEY" if os.name == "nt" else "key"


class ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_metadata_vars()
        tracking.active_user = None
        self.maxDiff = None

        self.model_config: ModelConfig = ModelConfig.from_dict(
            {
                "enabled": True,
                "materialized": "view",
                "persist_docs": {},
                "post-hook": [],
                "pre-hook": [],
                "vars": {},
                "quoting": {},
                "column_types": {},
                "tags": [],
            }
        )

        self.exposures: Dict[str, Exposure] = {
            "exposure.root.my_exposure": Exposure(
                name="my_exposure",
                type=ExposureType.Dashboard,
                owner=Owner(email="some@email.com"),
                resource_type=NodeType.Exposure,
                description="Test description",
                maturity=MaturityType.High,
                url="hhtp://mydashboard.com",
                depends_on=DependsOn(nodes=["model.root.multi"]),
                refs=[RefArgs(name="multi")],
                sources=[],
                fqn=["root", "my_exposure"],
                unique_id="exposure.root.my_exposure",
                package_name="root",
                path="my_exposure.sql",
                original_file_path="my_exposure.sql",
            )
        }

        self.metrics: Dict[str, Metric] = {
            "metric.root.my_metric": Metric(
                name="new_customers",
                label="New Customers",
                description="New customers",
                meta={"is_okr": True},
                tags=["okrs"],
                type=MetricType.SIMPLE,
                type_params=MetricTypeParams(
                    measure=MetricInputMeasure(
                        name="customers",
                        filter=WhereFilterIntersection(
                            [WhereFilter(where_sql_template="is_new = True")]
                        ),
                    )
                ),
                resource_type=NodeType.Metric,
                depends_on=DependsOn(nodes=["semantic_model.root.customers"]),
                refs=[RefArgs(name="customers")],
                fqn=["root", "my_metric"],
                unique_id="metric.root.my_metric",
                package_name="root",
                path="my_metric.yml",
                original_file_path="my_metric.yml",
            )
        }

        self.groups: Dict[str, Group] = {
            "group.root.my_group": Group(
                name="my_group",
                owner=Owner(email="some@email.com"),
                resource_type=NodeType.Group,
                unique_id="group.root.my_group",
                package_name="root",
                path="my_metric.yml",
                original_file_path="my_metric.yml",
            )
        }

        self.nested_nodes: Dict[str, ModelNode] = {
            "model.snowplow.events": ModelNode(
                name="events",
                database="dbt",
                schema="analytics",
                alias="events",
                resource_type=NodeType.Model,
                unique_id="model.snowplow.events",
                fqn=["snowplow", "events"],
                package_name="snowplow",
                refs=[],
                sources=[],
                metrics=[],
                depends_on=DependsOn(),
                config=self.model_config,
                tags=[],
                path="events.sql",
                original_file_path="events.sql",
                meta={},
                language="sql",
                raw_code="does not matter",
                checksum=FileHash.empty(),
            ),
            "model.root.events": ModelNode(
                name="events",
                database="dbt",
                schema="analytics",
                alias="events",
                resource_type=NodeType.Model,
                unique_id="model.root.events",
                fqn=["root", "events"],
                package_name="root",
                refs=[],
                sources=[],
                metrics=[],
                depends_on=DependsOn(),
                config=self.model_config,
                tags=[],
                path="events.sql",
                original_file_path="events.sql",
                meta={},
                language="sql",
                raw_code="does not matter",
                checksum=FileHash.empty(),
            ),
            "model.root.dep": ModelNode(
                name="dep",
                database="dbt",
                schema="analytics",
                alias="dep",
                resource_type=NodeType.Model,
                unique_id="model.root.dep",
                fqn=["root", "dep"],
                package_name="root",
                refs=[RefArgs(name="events")],
                sources=[],
                metrics=[],
                depends_on=DependsOn(nodes=["model.root.events"]),
                config=self.model_config,
                tags=[],
                path="multi.sql",
                original_file_path="multi.sql",
                meta={},
                language="sql",
                raw_code="does not matter",
                checksum=FileHash.empty(),
            ),
            "model.root.nested": ModelNode(
                name="nested",
                database="dbt",
                schema="analytics",
                alias="nested",
                resource_type=NodeType.Model,
                unique_id="model.root.nested",
                fqn=["root", "nested"],
                package_name="root",
                refs=[RefArgs(name="events")],
                sources=[],
                metrics=[],
                depends_on=DependsOn(nodes=["model.root.dep"]),
                config=self.model_config,
                tags=[],
                path="multi.sql",
                original_file_path="multi.sql",
                meta={},
                language="sql",
                raw_code="does not matter",
                checksum=FileHash.empty(),
            ),
            "model.root.sibling": ModelNode(
                name="sibling",
                database="dbt",
                schema="analytics",
                alias="sibling",
                resource_type=NodeType.Model,
                unique_id="model.root.sibling",
                fqn=["root", "sibling"],
                package_name="root",
                refs=[RefArgs(name="events")],
                sources=[],
                metrics=[],
                depends_on=DependsOn(nodes=["model.root.events"]),
                config=self.model_config,
                tags=[],
                path="multi.sql",
                original_file_path="multi.sql",
                meta={},
                language="sql",
                raw_code="does not matter",
                checksum=FileHash.empty(),
            ),
            "model.root.multi": ModelNode(
                name="multi",
                database="dbt",
                schema="analytics",
                alias="multi",
                resource_type=NodeType.Model,
                unique_id="model.root.multi",
                fqn=["root", "multi"],
                package_name="root",
                refs=[RefArgs(name="events")],
                sources=[],
                metrics=[],
                depends_on=DependsOn(nodes=["model.root.nested", "model.root.sibling"]),
                config=self.model_config,
                tags=[],
                path="multi.sql",
                original_file_path="multi.sql",
                meta={},
                language="sql",
                raw_code="does not matter",
                checksum=FileHash.empty(),
            ),
        }

        self.sources: Dict[str, SourceDefinition] = {
            "source.root.my_source.my_table": SourceDefinition(
                database="raw",
                schema="analytics",
                resource_type=NodeType.Source,
                identifier="some_source",
                name="my_table",
                source_name="my_source",
                source_description="My source description",
                description="Table description",
                loader="a_loader",
                unique_id="source.test.my_source.my_table",
                fqn=["test", "my_source", "my_table"],
                package_name="root",
                path="schema.yml",
                original_file_path="schema.yml",
            ),
        }

        self.semantic_models: Dict[str, Any] = {}
        self.saved_queries: Dict[str, Any] = {}

        for exposure in self.exposures.values():
            exposure.validate(exposure.to_dict(omit_none=True))
        for metric in self.metrics.values():
            metric.validate(metric.to_dict(omit_none=True))
        for node in self.nested_nodes.values():
            node.validate(node.to_dict(omit_none=True))
        for source in self.sources.values():
            source.validate(source.to_dict(omit_none=True))

        os.environ["DBT_ENV_CUSTOM_ENV_key"] = "value"

    def tearDown(self) -> None:
        del os.environ["DBT_ENV_CUSTOM_ENV_key"]
        reset_metadata_vars()

    @mock.patch.object(tracking, "active_user")
    @freezegun.freeze_time("2018-02-14T09:15:13Z")
    def test_no_nodes(self, mock_user: mock.MagicMock) -> None:
        manifest = Manifest(
            nodes={},
            sources={},
            macros={},
            docs={},
            disabled={},
            files={},
            exposures={},
            metrics={},
            selectors={},
            metadata=ManifestMetadata(generated_at=datetime.utcnow()),
            semantic_models={},
            saved_queries={},
        )

        invocation_id = dbt_common.invocation._INVOCATION_ID
        mock_user.id = "cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf"
        set_from_args(Namespace(SEND_ANONYMOUS_USAGE_STATS=False), None)
        self.assertEqual(
            manifest.writable_manifest().to_dict(omit_none=True),
            {
                "nodes": {},
                "sources": {},
                "macros": {},
                "exposures": {},
                "metrics": {},
                "groups": {},
                "selectors": {},
                "parent_map": {},
                "child_map": {},
                "group_map": {},
                "metadata": {
                    "generated_at": "2018-02-14T09:15:13Z",
                    "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json",
                    "dbt_version": dbt.version.__version__,
                    "env": {ENV_KEY_NAME: "value"},
                    "invocation_id": invocation_id,
                    "send_anonymous_usage_stats": False,
                    "user_id": "cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf",
                },
                "docs": {},
                "disabled": {},
                "semantic_models": {},
                "unit_tests": {},
                "saved_queries": {},
            },
        )

    @freezegun.freeze_time("2018-02-14T09:15:13Z")
    @mock.patch.object(tracking, "active_user")
    def test_nested_nodes(self, mock_user: mock.MagicMock) -> None:
        set_from_args(Namespace(SEND_ANONYMOUS_USAGE_STATS=False), None)
        mock_user.id = "cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf"
        nodes = deepcopy(self.nested_nodes)
        manifest = Manifest(
            nodes=nodes,
            sources={},
            macros={},
            docs={},
            disabled={},
            files={},
            exposures={},
            metrics={},
            selectors={},
            metadata=ManifestMetadata(generated_at=datetime.utcnow()),
        )
        serialized = manifest.writable_manifest().to_dict(omit_none=True)
        self.assertEqual(serialized["metadata"]["generated_at"], "2018-02-14T09:15:13Z")
        self.assertEqual(serialized["metadata"]["user_id"], mock_user.id)
        self.assertFalse(serialized["metadata"]["send_anonymous_usage_stats"])
        self.assertEqual(serialized["docs"], {})
        self.assertEqual(serialized["disabled"], {})
        parent_map = serialized["parent_map"]
        child_map = serialized["child_map"]
        self.assertEqual(set(parent_map), set(nodes))
        self.assertEqual(set(child_map), set(nodes))
        self.assertEqual(parent_map["model.root.sibling"], ["model.root.events"])
        self.assertEqual(parent_map["model.root.nested"], ["model.root.dep"])
        self.assertEqual(parent_map["model.root.dep"], ["model.root.events"])
        self.assertEqual(
            set(parent_map["model.root.multi"]), set(["model.root.nested", "model.root.sibling"])
        )
        self.assertEqual(parent_map["model.root.events"], [])
        self.assertEqual(parent_map["model.snowplow.events"], [])

        self.assertEqual(child_map["model.root.sibling"], ["model.root.multi"])
        self.assertEqual(child_map["model.root.nested"], ["model.root.multi"])
        self.assertEqual(child_map["model.root.dep"], ["model.root.nested"])
        self.assertEqual(child_map["model.root.multi"], [])
        self.assertEqual(
            set(child_map["model.root.events"]), set(["model.root.dep", "model.root.sibling"])
        )
        self.assertEqual(child_map["model.snowplow.events"], [])

    def test_build_flat_graph(self) -> None:
        exposures = deepcopy(self.exposures)
        metrics = deepcopy(self.metrics)
        groups = deepcopy(self.groups)
        nodes = deepcopy(self.nested_nodes)
        sources = deepcopy(self.sources)
        manifest = Manifest(
            nodes=nodes,
            sources=sources,
            macros={},
            docs={},
            disabled={},
            files={},
            exposures=exposures,
            metrics=metrics,
            groups=groups,
            selectors={},
        )
        manifest.build_flat_graph()
        flat_graph = manifest.flat_graph
        flat_exposures = flat_graph["exposures"]
        flat_groups = flat_graph["groups"]
        flat_metrics = flat_graph["metrics"]
        flat_nodes = flat_graph["nodes"]
        flat_sources = flat_graph["sources"]
        flat_semantic_models = flat_graph["semantic_models"]
        flat_saved_queries = flat_graph["saved_queries"]
        self.assertEqual(
            set(flat_graph),
            set(
                [
                    "exposures",
                    "groups",
                    "nodes",
                    "sources",
                    "metrics",
                    "semantic_models",
                    "saved_queries",
                ]
            ),
        )
        self.assertEqual(set(flat_exposures), set(self.exposures))
        self.assertEqual(set(flat_groups), set(self.groups))
        self.assertEqual(set(flat_metrics), set(self.metrics))
        self.assertEqual(set(flat_nodes), set(self.nested_nodes))
        self.assertEqual(set(flat_sources), set(self.sources))
        self.assertEqual(set(flat_semantic_models), set(self.semantic_models))
        self.assertEqual(set(flat_saved_queries), set(self.saved_queries))
        for node in flat_nodes.values():
            self.assertEqual(frozenset(node), REQUIRED_PARSED_NODE_KEYS)

    @mock.patch.object(tracking, "active_user")
    @freezegun.freeze_time("2018-02-14T09:15:13Z")
    def test_no_nodes_with_metadata(self, mock_user: mock.MagicMock) -> None:
        mock_user.id = "cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf"
        dbt_common.invocation._INVOCATION_ID = "01234567-0123-0123-0123-0123456789ab"
        set_from_args(Namespace(SEND_ANONYMOUS_USAGE_STATS=False), None)
        metadata = ManifestMetadata(
            project_id="098f6bcd4621d373cade4e832627b4f6",
            adapter_type="postgres",
            generated_at=datetime.utcnow(),
            user_id="cfc950