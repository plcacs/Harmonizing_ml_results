import dataclasses
import json
import os
import pickle
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import networkx as nx
import sqlparse
import dbt.tracking
from dbt.adapters.factory import get_adapter
from dbt.clients import jinja
from dbt.context.providers import generate_runtime_model_context, generate_runtime_unit_test_context
from dbt.contracts.graph.manifest import Manifest, UniqueID
from dbt.contracts.graph.nodes import GenericTestNode, GraphMemberNode, InjectedCTE, ManifestNode, ManifestSQLNode, ModelNode, SeedNode, UnitTestDefinition, UnitTestNode
from dbt.events.types import FoundStats, WritingInjectedSQLForNode
from dbt.exceptions import DbtInternalError, DbtRuntimeError, ForeignKeyConstraintToSyntaxError, GraphDependencyNotFoundError, ParsingError
from dbt.flags import get_flags
from dbt.graph import Graph
from dbt.node_types import ModelLanguage, NodeType
from dbt_common.clients.system import make_directory
from dbt_common.contracts.constraints import ConstraintType
from dbt_common.events.contextvars import get_node_info
from dbt_common.events.format import pluralize
from dbt_common.events.functions import fire_event
from dbt_common.invocation import get_invocation_id

class SeenDetails:
    def __init__(self, node_id: str):
        self.visits: int = 0
        self.ancestors: set = set()
        self.awaits_tests: set = set()

class Linker:
    def __init__(self, data: Optional[Dict] = None):
        if data is None:
            data = {}
        self.graph: nx.DiGraph = nx.DiGraph(**data)

    # ... (rest of the code remains the same)

class Compiler:
    def __init__(self, config: Any):
        self.config: Any = config

    # ... (rest of the code remains the same)

def inject_ctes_into_sql(sql: str, ctes: List[InjectedCTE]) -> str:
    # ... (rest of the code remains the same)
