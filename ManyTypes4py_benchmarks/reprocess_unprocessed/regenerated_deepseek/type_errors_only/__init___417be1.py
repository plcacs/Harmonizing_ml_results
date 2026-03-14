"""
Type inference of Python code in |jedi| is based on three assumptions:

* The code uses as least side effects as possible. Jedi understands certain
  list/tuple/set modifications, but there's no guarantee that Jedi detects
  everything (list.append in different modules for example).
* No magic is being used:

  - metaclasses
  - ``setattr()`` / ``__import__()``
  - writing to ``globals()``, ``locals()``, ``object.__dict__``
* The programmer is not a total dick, e.g. like `this
  <https://github.com/davidhalter/jedi/issues/24>`_ :-)

The actual algorithm is based on a principle I call lazy type inference.  That
said, the typical entry point for static analysis is calling
``infer_expr_stmt``. There's separate logic for autocompletion in the API, the
inference_state is all about inferring an expression.

TODO this paragraph is not what jedi does anymore, it's similar, but not the
same.

Now you need to understand what follows after ``infer_expr_stmt``. Let's
make an example::

    import datetime
    datetime.date.toda# <-- cursor here

First of all, this module doesn't care about completion. It really just cares
about ``datetime.date``. At the end of the procedure ``infer_expr_stmt`` will
return the ``date`` class.

To *visualize* this (simplified):

- ``InferenceState.infer_expr_stmt`` doesn't do much, because there's no assignment.
- ``Context.infer_node`` cares for resolving the dotted path
- ``InferenceState.find_types`` searches for global definitions of datetime, which
  it finds in the definition of an import, by scanning the syntax tree.
- Using the import logic, the datetime module is found.
- Now ``find_types`` is called again by ``infer_node`` to find ``date``
  inside the datetime module.

Now what would happen if we wanted ``datetime.date.foo.bar``? Two more
calls to ``find_types``. However the second call would be ignored, because the
first one would return nothing (there's no foo attribute in ``date``).

What if the import would contain another ``ExprStmt`` like this::

    from foo import bar
    Date = bar.baz

Well... You get it. Just another ``infer_expr_stmt`` recursion. It's really
easy. Python can obviously get way more complicated then this. To understand
tuple assignments, list comprehensions and everything else, a lot more code had
to be written.

Jedi has been tested very well, so you can just start modifying code. It's best
to write your own test first for your "new" feature. Don't be scared of
breaking stuff. As long as the tests pass, you're most likely to be fine.

I need to mention now that lazy type inference is really good because it
only *inferes* what needs to be *inferred*. All the statements and modules
that are not used are just being ignored.
"""
import parso
from jedi.file_io import FileIO
from jedi import debug
from jedi import settings
from jedi.inference import imports
from jedi.inference import recursion
from jedi.inference.cache import inference_state_function_cache
from jedi.inference import helpers
from jedi.inference.names import TreeNameDefinition
from jedi.inference.base_value import ContextualizedNode, ValueSet, iterate_values
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.syntax_tree import infer_expr_stmt, check_tuple_assignments, tree_name_to_values
from jedi.inference.imports import follow_error_node_imports_if_possible
from jedi.plugins import plugin_manager
from typing import Optional, Tuple, Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from jedi.project import Project
    from parso.python.tree import Name
    from parso.tree import NodeOrLeaf
    from jedi.inference.base_value import Context, Value
    from jedi.inference.value import ModuleValue
    from jedi.api.environment import Environment
    from parso.grammar import Grammar
    from jedi.inference.imports import ModuleCache
    from jedi.inference.arguments import Arguments


class InferenceState:

    def __init__(self, project: 'Project', environment: Optional['Environment'] = None, script_path: Optional[str] = None) -> None:
        if environment is None:
            environment = project.get_environment()
        self.environment: 'Environment' = environment
        self.script_path: Optional[str] = script_path
        self.compiled_subprocess: Any = environment.get_inference_state_subprocess(self)
        self.grammar: 'Grammar' = environment.get_grammar()
        self.latest_grammar: 'Grammar' = parso.load_grammar(version='3.7')
        self.memoize_cache: Dict[Any, Any] = {}
        self.module_cache: 'ModuleCache' = imports.ModuleCache()
        self.stub_module_cache: Dict[Any, Any] = {}
        self.compiled_cache: Dict[Any, Any] = {}
        self.inferred_element_counts: Dict[Any, Any] = {}
        self.mixed_cache: Dict[Any, Any] = {}
        self.analysis: List[Any] = []
        self.dynamic_params_depth: int = 0
        self.is_analysis: bool = False
        self.project: 'Project' = project
        self.access_cache: Dict[Any, Any] = {}
        self.allow_descriptor_getattr: bool = False
        self.flow_analysis_enabled: bool = True
        self.reset_recursion_limitations()

    def import_module(self, import_names: Tuple[str, ...], sys_path: Optional[Tuple[str, ...]] = None, prefer_stubs: bool = True) -> 'ValueSet':
        return imports.import_module_by_names(self, import_names, sys_path, prefer_stubs=prefer_stubs)

    @staticmethod
    @plugin_manager.decorate()
    def execute(value: 'Value', arguments: 'Arguments') -> 'ValueSet':
        debug.dbg('execute: %s %s', value, arguments)
        with debug.increase_indent_cm():
            value_set: 'ValueSet' = value.py__call__(arguments=arguments)
        debug.dbg('execute result: %s in %s', value_set, value)
        return value_set

    @property
    @inference_state_function_cache()
    def builtins_module(self) -> 'ModuleValue':
        module_name = 'builtins'
        builtins_module: 'ModuleValue'
        builtins_module, = self.import_module((module_name,), sys_path=())
        return builtins_module

    @property
    @inference_state_function_cache()
    def typing_module(self) -> 'ModuleValue':
        typing_module: 'ModuleValue'
        typing_module, = self.import_module(('typing',))
        return typing_module

    def reset_recursion_limitations(self) -> None:
        self.recursion_detector: recursion.RecursionDetector = recursion.RecursionDetector()
        self.execution_recursion_detector: recursion.ExecutionRecursionDetector = recursion.ExecutionRecursionDetector(self)

    def get_sys_path(self, **kwargs: Any) -> List[str]:
        """Convenience function"""
        return self.project._get_sys_path(self, **kwargs)

    def infer(self, context: 'Context', name: 'Name') -> 'ValueSet':
        def_: Optional['NodeOrLeaf'] = name.get_definition(import_name_always=True)
        if def_ is not None:
            type_: str = def_.type
            is_classdef: bool = type_ == 'classdef'
            if is_classdef or type_ == 'funcdef':
                if is_classdef:
                    c: 'ClassValue' = ClassValue(self, context, name.parent)
                else:
                    c = FunctionValue.from_context(context, name.parent)
                return ValueSet([c])
            if type_ == 'expr_stmt':
                is_simple_name: bool = name.parent.type not in ('power', 'trailer')
                if is_simple_name:
                    return infer_expr_stmt(context, def_, name)
            if type_ == 'for_stmt':
                container_types: 'ValueSet' = context.infer_node(def_.children[3])
                cn: ContextualizedNode = ContextualizedNode(context, def_.children[3])
                for_types: 'ValueSet' = iterate_values(container_types, cn)
                n: TreeNameDefinition = TreeNameDefinition(context, name)
                return check_tuple_assignments(n, for_types)
            if type_ in ('import_from', 'import_name'):
                return imports.infer_import(context, name)
            if type_ == 'with_stmt':
                return tree_name_to_values(self, context, name)
            elif type_ == 'param':
                return context.py__getattribute__(name.value, position=name.end_pos)
            elif type_ == 'namedexpr_test':
                return context.infer_node(def_)
        else:
            result: Optional['ValueSet'] = follow_error_node_imports_if_possible(context, name)
            if result is not None:
                return result
        return helpers.infer_call_of_leaf(context, name)

    def parse_and_get_code(self, code: Optional[str] = None, path: Optional[str] = None, use_latest_grammar: bool = False, file_io: Optional[FileIO] = None, **kwargs: Any) -> Tuple[Any, str]:
        if code is None:
            if file_io is None:
                file_io = FileIO(path)
            code = file_io.read()
        code = parso.python_bytes_to_unicode(code, encoding='utf-8', errors='replace')
        if len(code) > settings._cropped_file_size:
            code = code[:settings._cropped_file_size]
        grammar: 'Grammar' = self.latest_grammar if use_latest_grammar else self.grammar
        return (grammar.parse(code=code, path=path, file_io=file_io, **kwargs), code)

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        return self.parse_and_get_code(*args, **kwargs)[0]