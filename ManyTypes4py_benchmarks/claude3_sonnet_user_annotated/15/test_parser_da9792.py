import os
import unittest
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest import mock

import yaml

from dbt import tracking
from dbt.artifacts.resources import ModelConfig, RefArgs
from dbt.artifacts.resources.v1.model import (
    ModelBuildAfter,
    ModelFreshnessDependsOnOptions,
)
from dbt.context.context_config import ContextConfig
from dbt.contracts.files import FileHash, FilePath, SchemaSourceFile, SourceFile
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.model_config import NodeConfig, SnapshotConfig, TestConfig
from dbt.contracts.graph.nodes import (
    AnalysisNode,
    DependsOn,
    Macro,
    ModelNode,
    SingularTestNode,
    SnapshotNode,
    UnpatchedSourceDefinition,
)
from dbt.exceptions import CompilationError, ParsingError, SchemaConfigError
from dbt.flags import set_from_args
from dbt.node_types import NodeType
from dbt.parser import (
    AnalysisParser,
    GenericTestParser,
    MacroParser,
    ModelParser,
    SchemaParser,
    SingularTestParser,
    SnapshotParser,
)
from dbt.parser.common import YamlBlock
from dbt.parser.models import (
    _get_config_call_dict,
    _get_exp_sample_result,
    _get_sample_result,
    _get_stable_sample_result,
    _shift_sources,
)
from dbt.parser.schemas import (
    AnalysisPatchParser,
    MacroPatchParser,
    ModelPatchParser,
    SourceParser,
    TestablePatchParser,
    yaml_from_file,
)
from dbt.parser.search import FileBlock
from dbt.parser.sources import SourcePatcher
from tests.unit.utils import (
    MockNode,
    config_from_parts_or_dicts,
    generate_name_macros,
    normalize,
)
