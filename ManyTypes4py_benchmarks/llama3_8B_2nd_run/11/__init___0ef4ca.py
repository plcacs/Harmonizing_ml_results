from __future__ import annotations
import importlib
import inspect
import logging
import os
import sys
import typing
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable
if TYPE_CHECKING:
    from collections import OrderedDict
    from IPython.core.interactiveshell import InteractiveShell
from IPython.core.getipython import get_ipython
from IPython.core.magic import needs_local_scope, register_line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
try:
    import rich.console as rich_console
    import rich.syntax as rich_syntax
except ImportError:
    pass
from kedro.framework.cli import load_entry_points
from kedro.framework.cli.project import CONF_SOURCE_HELP, PARAMS_ARG_HELP
from kedro.framework.cli.utils import ENV_HELP, _split_params
from kedro.framework.project import LOGGING, _ProjectPipelines, configure_project, pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.pipeline.node import Node
from kedro.utils import _find_kedro_project, _is_databricks

logger: logging.Logger = logging.getLogger(__name__)
FunctionParameters: MappingProxyType
RICH_INSTALLED: bool = True if importlib.util.find_spec('rich') is not None else False

def load_ipython_extension(ipython: InteractiveShell) -> None:
    ...

@needs_local_scope
@magic_arguments()
@argument('path', type=str, help='Path to the project root directory. If not given, use the previously set project root.', nargs='?', default=None)
@argument('-e', '--env', type=str, default=None, help=ENV_HELP)
@argument('--params', type=lambda value: _split_params(None, None, value), default=None, help=PARAMS_ARG_HELP)
@argument('--conf-source', type=str, default=None, help=CONF_SOURCE_HELP)
def magic_reload_kedro(line: str, local_ns: dict | None, conf_source: str | None) -> None:
    ...

def reload_kedro(path: str | None, env: str | None, extra_params: tuple[str, ...] | None, local_namespace: dict | None, conf_source: str | None) -> None:
    ...

def _resolve_project_path(path: str | None, local_namespace: dict | None) -> Path:
    ...

def _remove_cached_modules(package_name: str) -> None:
    ...

@typing.no_type_check
@magic_arguments()
@argument('node', type=str, help='Name of the Node.', nargs='?', default=None)
def magic_load_node(args: argparse.Namespace) -> None:
    ...

class _NodeBoundArguments(inspect.BoundArguments):
    ...

def _create_cell_with_text(text: str, is_jupyter: bool) -> None:
    ...

def _print_cells(cells: list[str]) -> None:
    ...

def _load_node(node_name: str, pipelines: dict[str, Node]) -> list[str]:
    ...

def _find_node(node_name: str, pipelines: dict[str, Node]) -> Node:
    ...

def _prepare_imports(node_func: Callable) -> str:
    ...

def _get_node_bound_arguments(node: Node) -> _NodeBoundArguments:
    ...

def _prepare_node_inputs(node_bound_arguments: _NodeBoundArguments) -> dict[str, str]:
    ...

def _format_node_inputs_text(input_params_dict: dict[str, str]) -> str | None:
    ...

def _prepare_function_body(func: Callable) -> str:
    ...

def _prepare_function_call(node_func: Callable, node_bound_arguments: _NodeBoundArguments) -> str:
    ...
