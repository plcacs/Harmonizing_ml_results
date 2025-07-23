"""
This script creates an IPython extension to load Kedro-related variables in
local scope.
"""
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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
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
logger = logging.getLogger(__name__)
FunctionParameters = MappingProxyType
RICH_INSTALLED: bool = True if importlib.util.find_spec('rich') is not None else False

def load_ipython_extension(ipython: 'InteractiveShell') -> None:
    """
    Main entry point when %load_ext kedro.ipython is executed, either manually or
    automatically through `kedro ipython` or `kedro jupyter lab/notebook`.
    IPython will look for this function specifically.
    See https://ipython.readthedocs.io/en/stable/config/extensions/index.html
    """
    ipython.register_magic_function(func=magic_reload_kedro, magic_name='reload_kedro')
    logger.info("Registered line magic '%reload_kedro'")
    ipython.register_magic_function(func=magic_load_node, magic_name='load_node')
    logger.info("Registered line magic '%load_node'")
    if _find_kedro_project(Path.cwd()) is None:
        logger.warning("Kedro extension was registered but couldn't find a Kedro project. Make sure you run '%reload_kedro <project_root>'.")
        return
    reload_kedro()

@typing.no_type_check
@needs_local_scope
@magic_arguments()
@argument('path', type=str, help='Path to the project root directory. If not given, use the previously setproject root.', nargs='?', default=None)
@argument('-e', '--env', type=str, default=None, help=ENV_HELP)
@argument('--params', type=lambda value: _split_params(None, None, value), default=None, help=PARAMS_ARG_HELP)
@argument('--conf-source', type=str, default=None, help=CONF_SOURCE_HELP)
def magic_reload_kedro(line: str, local_ns: Optional[Dict[str, Any]] = None, conf_source: Optional[str] = None) -> None:
    """
    The `%reload_kedro` IPython line magic.
    See https://docs.kedro.org/en/stable/notebooks_and_ipython/kedro_and_notebooks.html#reload-kedro-line-magic
    for more.
    """
    args = parse_argstring(magic_reload_kedro, line)
    reload_kedro(args.path, args.env, args.params, local_ns, args.conf_source)

def reload_kedro(path: Optional[str] = None, env: Optional[str] = None, 
                extra_params: Optional[Dict[str, Any]] = None, 
                local_namespace: Optional[Dict[str, Any]] = None, 
                conf_source: Optional[str] = None) -> None:
    """Function that underlies the %reload_kedro Line magic. This should not be imported
    or run directly but instead invoked through %reload_kedro."""
    project_path = _resolve_project_path(path, local_namespace)
    metadata = bootstrap_project(project_path)
    _remove_cached_modules(metadata.package_name)
    configure_project(metadata.package_name)
    session = KedroSession.create(project_path, env=env, extra_params=extra_params, conf_source=conf_source)
    context = session.load_context()
    catalog = context.catalog
    get_ipython().push(variables={'context': context, 'catalog': catalog, 'session': session, 'pipelines': pipelines})
    logger.info('Kedro project %s', str(metadata.project_name))
    logger.info("Defined global variable 'context', 'session', 'catalog' and 'pipelines'")
    for line_magic in load_entry_points('line_magic'):
        register_line_magic(needs_local_scope(line_magic))
        logger.info("Registered line magic '%s'", line_magic.__name__)

def _resolve_project_path(path: Optional[str] = None, 
                         local_namespace: Optional[Dict[str, Any]] = None) -> Path:
    """
    Resolve the project path to use with reload_kedro, updating or adding it
    (in-place) to the local ipython Namespace (``local_namespace``) if necessary.

    Arguments:
        path: the path to use as a string object
        local_namespace: Namespace with local variables of the scope where the line
            magic is invoked in a dict.
    """
    if path:
        project_path = Path(path).expanduser().resolve()
    else:
        if local_namespace and local_namespace.get('context') and hasattr(local_namespace['context'], 'project_path'):
            project_path = local_namespace['context'].project_path
        else:
            project_path = _find_kedro_project(Path.cwd())
        if project_path:
            logger.info("Resolved project path as: %s.\nTo set a different path, run '%%reload_kedro <project_root>'", project_path)
    if project_path and local_namespace and local_namespace.get('context') and hasattr(local_namespace['context'], 'project_path') and (project_path != local_namespace['context'].project_path):
        logger.info('Updating path to Kedro project: %s...', project_path)
    return project_path

def _remove_cached_modules(package_name: str) -> None:
    to_remove: List[str] = [mod for mod in sys.modules if mod.startswith(package_name)]
    for module in to_remove:
        del sys.modules[module]

def _guess_run_environment() -> str:
    """Best effort to guess the IPython/Jupyter environment"""
    if os.environ.get('VSCODE_PID') or os.environ.get('VSCODE_CWD'):
        return 'vscode'
    elif _is_databricks():
        return 'databricks'
    elif hasattr(get_ipython(), 'kernel'):
        return 'jupyter'
    else:
        return 'ipython'

@typing.no_type_check
@magic_arguments()
@argument('node', type=str, help='Name of the Node.', nargs='?', default=None)
def magic_load_node(args: str) -> None:
    """The line magic %load_node <node_name>.
    Currently, this feature is only available for Jupyter Notebook (>7.0), Jupyter Lab, IPython,
    and VSCode Notebook. This line magic will generate code in multiple cells to load
    datasets from `DataCatalog`, import relevant functions and modules, node function
    definition and a function call. If generating code is not possible, it will print
    the code instead.
    """
    parameters = parse_argstring(magic_load_node, args)
    node_name = parameters.node
    cells = _load_node(node_name, pipelines)
    run_environment = _guess_run_environment()
    if run_environment == 'jupyter':
        for cell in cells:
            _create_cell_with_text(cell, is_jupyter=True)
    elif run_environment in ('ipython', 'vscode'):
        combined_cell = '\n\n'.join(cells)
        _create_cell_with_text(combined_cell, is_jupyter=False)
    else:
        _print_cells(cells)

class _NodeBoundArguments(inspect.BoundArguments):
    """Similar to inspect.BoundArguments"""

    def __init__(self, signature: inspect.Signature, arguments: Dict[str, Any]):
        super().__init__(signature, arguments)

    @property
    def input_params_dict(self) -> Dict[str, Any]:
        """A mapping of {variable name: dataset_name}"""
        var_positional_arg_name = self._find_var_positional_arg()
        inputs_params_dict: Dict[str, Any] = {}
        for param, dataset_name in self.arguments.items():
            if param == var_positional_arg_name:
                for arg in dataset_name:
                    inputs_params_dict[arg] = arg
            else:
                inputs_params_dict[param] = dataset_name
        return inputs_params_dict

    def _find_var_positional_arg(self) -> Optional[str]:
        """Find the name of the VAR_POSITIONAL argument( *args), if any."""
        for k, v in self.signature.parameters.items():
            if v.kind == inspect.Parameter.VAR_POSITIONAL:
                return k
        return None

def _create_cell_with_text(text: str, is_jupyter: bool = True) -> None:
    if is_jupyter:
        from ipylab import JupyterFrontEnd
        app = JupyterFrontEnd()
        app.commands.execute('notebook:insert-cell-below')
        app.commands.execute('notebook:replace-selection', {'text': text})
    else:
        get_ipython().set_next_input(text)

def _print_cells(cells: List[str]) -> None:
    for cell in cells:
        if RICH_INSTALLED is True:
            rich_console.Console().print('')
            rich_console.Console().print(rich_syntax.Syntax(cell, 'python', theme='monokai', line_numbers=False))
        else:
            print('')
            print(cell)

def _load_node(node_name: str, pipelines: _ProjectPipelines) -> List[str]:
    """Prepare the code to load dataset from catalog, import statements and function body.

    Args:
        node_name (str): The name of the node.

    Returns:
        list[str]: A list of string which is the generated code, each string represent a
        notebook cell.
    """
    warnings.warn('This is an experimental feature, only Jupyter Notebook (>7.0), Jupyter Lab, IPython, and VSCode Notebook are supported. If you encounter unexpected behaviour or would like to suggest feature enhancements, add it under this github issue https://github.com/kedro-org/kedro/issues/3580')
    node = _find_node(node_name, pipelines)
    node_func = node.func
    imports_cell = _prepare_imports(node_func)
    function_definition_cell = _prepare_function_body(node_func)
    node_bound_arguments = _get_node_bound_arguments(node)
    inputs_params_mapping = _prepare_node_inputs(node_bound_arguments)
    node_inputs_cell = _format_node_inputs_text(inputs_params_mapping)
    function_call_cell = _prepare_function_call(node_func, node_bound_arguments)
    cells: List[str] = []
    if node_inputs_cell:
        cells.append(node_inputs_cell)
    cells.append(imports_cell)
    cells.append(function_definition_cell)
    cells.append(function_call_cell)
    return cells

def _find_node(node_name: str, pipelines: _ProjectPipelines) -> Node:
    for pipeline in pipelines.values():
        try:
            found_node = pipeline.filter(node_names=[node_name]).nodes[0]
            return found_node
        except ValueError:
            continue
    raise ValueError(f"Node with name='{node_name}' not found in any pipelines. Remember to specify the node name, not the node function.")

def _prepare_imports(node_func: Callable) -> str:
    """Prepare the import statements for loading a node."""
    python_file = inspect.getsourcefile(node_func)
    logger.info(f'Loading node definition from {python_file}')
    if python_file:
        import_statement: List[str] = []
        with open(python_file) as file:
            inside_bracket = False
            for _ in file.readlines():
                line = _.strip()
                if not inside_bracket:
                    if line.startswith('from') or line.startswith('import'):
                        import_statement.append(line)
                        if line.endswith('('):
                            inside_bracket = True
                else:
                    import_statement.append(line)
                    if line.endswith(')'):
                        inside_bracket = False
        clean_imports = '\n'.join(import_statement).strip()
        return clean_imports
    else:
        raise FileNotFoundError(f'Could not find {node_func.__name__}')

def _get_node_bound_arguments(node: Node) -> _NodeBoundArguments:
    node_func = node.func
    node_inputs = node.inputs
    args, kwargs = Node._process_inputs_for_bind(node_inputs)
    signature = inspect.signature(node_func)
    bound_arguments = signature.bind(*args, **kwargs)
    return _NodeBoundArguments(bound_arguments.signature, bound_arguments.arguments)

def _prepare_node_inputs(node_bound_arguments: _NodeBoundArguments) -> Dict[str, Any]:
    return node_bound_arguments.input_params_dict

def _format_node_inputs_text(input_params_dict: Dict[str, Any]) -> Optional[str]:
    statements: List[str] = ['# Prepare necessary inputs for debugging', '# All debugging inputs must be defined in your project catalog']
    if not input_params_dict:
        return None
    for func_param, dataset_name in input_params_dict.items():
        statements.append(f'{func_param} = catalog.load("{dataset_name}")')
    input_statements = '\n'.join(statements)
    return input_statements

def _prepare_function_body(func: Callable) -> str:
    source_lines, _ = inspect.getsourcelines(func)
    body = ''.join(source_lines)
    return body

def _prepare_function_call(node_func: Callable, node_bound_arguments: _NodeBoundArguments) -> str:
    """Prepare the text for the function call."""
    func_name = node_func.__name__
    args = node_bound_arguments.input_params_dict
    kwargs = node_bound_arguments.kwargs
    args_str_literal: List[str] = [f'{node_input}' for node_input in args] if args else []
    kwargs_str_literal: List[str] = [f'{node_input}={dataset_name}' for node_input, dataset_name in kwargs.items()]
    func_params = ', '.join(args_str_literal + kwargs_str_literal)
    body = f'{func_name}({func_params})'
    return body
