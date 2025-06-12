import re
from textwrap import dedent
from inspect import Parameter
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Set, Tuple, Union
from parso.python.token import PythonTokenTypes
from parso.python import tree
from parso.tree import search_ancestor, Leaf
from parso import split_lines
from jedi import debug
from jedi import settings
from jedi.api import classes
from jedi.api import helpers
from jedi.api import keywords
from jedi.api.strings import complete_dict
from jedi.api.file_name import complete_file_name
from jedi.inference import imports
from jedi.inference.base_value import ValueSet
from jedi.inference.helpers import infer_call_of_leaf, parse_dotted_names
from jedi.inference.context import get_global_filters
from jedi.inference.value import TreeInstance
from jedi.inference.docstring_utils import DocstringModule
from jedi.inference.names import ParamNameWrapper, SubModuleName
from jedi.inference.gradual.conversion import convert_values, convert_names
from jedi.parser_utils import cut_value_at_position
from jedi.plugins import plugin_manager
from jedi.inference.base_value import Value
from jedi.inference.context import ModuleContext
from jedi.inference.value import ModuleValue
from jedi.inference.names import AbstractNameDefinition
from jedi.api.classes import Completion, Name
from jedi.inference.signature import AbstractSignature
from jedi.inference.gradual.stub_value import StubModuleValue

class ParamNameWithEquals(ParamNameWrapper):
    def get_public_name(self) -> str:
        return self.string_name + '='

def _get_signature_param_names(
    signatures: List[AbstractSignature],
    positional_count: int,
    used_kwargs: List[str]
) -> Generator[ParamNameWithEquals, None, None]:
    for call_sig in signatures:
        for i, p in enumerate(call_sig.params):
            kind = p.kind
            if i < positional_count and kind == Parameter.POSITIONAL_OR_KEYWORD:
                continue
            if kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY) and p.name not in used_kwargs:
                yield ParamNameWithEquals(p._name)

def _must_be_kwarg(
    signatures: List[AbstractSignature],
    positional_count: int,
    used_kwargs: List[str]
) -> bool:
    if used_kwargs:
        return True
    must_be_kwarg = True
    for signature in signatures:
        for i, p in enumerate(signature.params):
            kind = p.kind
            if kind is Parameter.VAR_POSITIONAL:
                return False
            if i >= positional_count and kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY):
                must_be_kwarg = False
                break
        if not must_be_kwarg:
            break
    return must_be_kwarg

def filter_names(
    inference_state: Any,
    completion_names: List[AbstractNameDefinition],
    stack: Any,
    like_name: str,
    fuzzy: bool,
    cached_name: Optional[str]
) -> Generator[Completion, None, None]:
    comp_dct = set()
    if settings.case_insensitive_completion:
        like_name = like_name.lower()
    for name in completion_names:
        string = name.string_name
        if settings.case_insensitive_completion:
            string = string.lower()
        if helpers.match(string, like_name, fuzzy=fuzzy):
            new = classes.Completion(inference_state, name, stack, len(like_name), is_fuzzy=fuzzy, cached_name=cached_name)
            k = (new.name, new.complete)
            if k not in comp_dct:
                comp_dct.add(k)
                tree_name = name.tree_name
                if tree_name is not None:
                    definition = tree_name.get_definition()
                    if definition is not None and definition.type == 'del_stmt':
                        continue
                yield new

def _remove_duplicates(
    completions: List[Completion],
    other_completions: Iterable[Completion]
) -> List[Completion]:
    names = {d.name for d in other_completions}
    return [c for c in completions if c.name not in names]

def get_user_context(
    module_context: ModuleContext,
    position: Tuple[int, int]
) -> Any:
    leaf = module_context.tree_node.get_leaf_for_position(position, include_prefixes=True)
    return module_context.create_context(leaf)

def get_flow_scope_node(
    module_node: tree.Module,
    position: Tuple[int, int]
) -> Union[tree.Scope, tree.Flow]:
    node = module_node.get_leaf_for_position(position, include_prefixes=True)
    while not isinstance(node, (tree.Scope, tree.Flow)):
        node = node.parent
    return node

@plugin_manager.decorate()
def complete_param_names(
    context: Any,
    function_name: str,
    decorator_nodes: List[tree.Node]
) -> List[AbstractNameDefinition]:
    return []

class Completion:
    def __init__(
        self,
        inference_state: Any,
        module_context: ModuleContext,
        code_lines: List[str],
        position: Tuple[int, int],
        signatures_callback: Any,
        fuzzy: bool = False
    ):
        self._inference_state = inference_state
        self._module_context = module_context
        self._module_node = module_context.tree_node
        self._code_lines = code_lines
        self._like_name = helpers.get_on_completion_name(self._module_node, code_lines, position)
        self._original_position = position
        self._signatures_callback = signatures_callback
        self._fuzzy = fuzzy

    def complete(self) -> List[Completion]:
        leaf = self._module_node.get_leaf_for_position(self._original_position, include_prefixes=True)
        string, start_leaf, quote = _extract_string_while_in_string(leaf, self._original_position)
        prefixed_completions = complete_dict(self._module_context, self._code_lines, start_leaf or leaf, self._original_position, None if string is None else quote + string, fuzzy=self._fuzzy)
        if string is not None and (not prefixed_completions):
            prefixed_completions = list(complete_file_name(self._inference_state, self._module_context, start_leaf, quote, string, self._like_name, self._signatures_callback, self._code_lines, self._original_position, self._fuzzy))
        if string is not None:
            if not prefixed_completions and '\n' in string:
                prefixed_completions = self._complete_in_string(start_leaf, string)
            return prefixed_completions
        cached_name, completion_names = self._complete_python(leaf)
        completions = list(filter_names(self._inference_state, completion_names, self.stack, self._like_name, self._fuzzy, cached_name=cached_name))
        return _remove_duplicates(prefixed_completions, completions) + sorted(completions, key=lambda x: (x.name.startswith('__'), x.name.startswith('_'), x.name.lower()))

    def _complete_python(self, leaf: Leaf) -> Tuple[Optional[str], List[AbstractNameDefinition]]:
        grammar = self._inference_state.grammar
        self.stack = stack = None
        self._position = (self._original_position[0], self._original_position[1] - len(self._like_name))
        cached_name = None
        try:
            self.stack = stack = helpers.get_stack_at_position(grammar, self._code_lines, leaf, self._position)
        except helpers.OnErrorLeaf as e:
            value = e.error_leaf.value
            if value == '.':
                return (cached_name, [])
            return (cached_name, self._complete_global_scope())
        allowed_transitions = list(stack._allowed_transition_names_and_token_types())
        if 'if' in allowed_transitions:
            leaf = self._module_node.get_leaf_for_position(self._position, include_prefixes=True)
            previous_leaf = leaf.get_previous_leaf()
            indent = self._position[1]
            if not leaf.start_pos <= self._position <= leaf.end_pos:
                indent = leaf.start_pos[1]
            if previous_leaf is not None:
                stmt = previous_leaf
                while True:
                    stmt = search_ancestor(stmt, 'if_stmt', 'for_stmt', 'while_stmt', 'try_stmt', 'error_node')
                    if stmt is None:
                        break
                    type_ = stmt.type
                    if type_ == 'error_node':
                        first = stmt.children[0]
                        if isinstance(first, Leaf):
                            type_ = first.value + '_stmt'
                    if stmt.start_pos[1] == indent:
                        if type_ == 'if_stmt':
                            allowed_transitions += ['elif', 'else']
                        elif type_ == 'try_stmt':
                            allowed_transitions += ['except', 'finally', 'else']
                        elif type_ == 'for_stmt':
                            allowed_transitions.append('else')
        completion_names = []
        kwargs_only = False
        if any((t in allowed_transitions for t in (PythonTokenTypes.NAME, PythonTokenTypes.INDENT))):
            nonterminals = [stack_node.nonterminal for stack_node in stack]
            nodes = _gather_nodes(stack)
            if nodes and nodes[-1] in ('as', 'def', 'class'):
                return (cached_name, list(self._complete_inherited(is_function=True)))
            elif 'import_stmt' in nonterminals:
                level, names = parse_dotted_names(nodes, 'import_from' in nonterminals)
                only_modules = not ('import_from' in nonterminals and 'import' in nodes)
                completion_names += self._get_importer_names(names, level, only_modules=only_modules)
            elif nonterminals[-1] in ('trailer', 'dotted_name') and nodes[-1] == '.':
                dot = self._module_node.get_leaf_for_position(self._position)
                if dot.type == 'endmarker':
                    dot = leaf.get_previous_leaf()
                cached_name, n = self._complete_trailer(dot.get_previous_leaf())
                completion_names += n
            elif self._is_parameter_completion():
                completion_names += self._complete_params(leaf)
            else:
                if nodes[-1] in ['(', ','] and nonterminals[-1] in ('trailer', 'arglist', 'decorator'):
                    signatures = self._signatures_callback(*self._position)
                    if signatures:
                        call_details = signatures[0]._call_details
                        used_kwargs = list(call_details.iter_used_keyword_arguments())
                        positional_count = call_details.count_positional_arguments()
                        completion_names += _get_signature_param_names(signatures, positional_count, used_kwargs)
                        kwargs_only = _must_be_kwarg(signatures, positional_count, used_kwargs)
                if not kwargs_only:
                    completion_names += self._complete_global_scope()
                    completion_names += self._complete_inherited(is_function=False)
        if not kwargs_only:
            current_line = self._code_lines[self._position[0] - 1][:self._position[1]]
            completion_names += self._complete_keywords(allowed_transitions, only_values=not (not current_line or (current_line[-1] in ' \t.;' and current_line[-3:] != '...')))
        return (cached_name, completion_names)

    def _is_parameter_completion(self) -> bool:
        tos = self.stack[-1]
        if tos.nonterminal == 'lambdef' and len(tos.nodes) == 1:
            return True
        if tos.nonterminal in 'parameters':
            return True
        return tos.nonterminal in ('typedargslist', 'varargslist') and tos.nodes[-1] == ','

    def _complete_params(self, leaf: Leaf) -> List[AbstractNameDefinition]:
        stack_node = self.stack[-2]
        if stack_node.nonterminal == 'parameters':
            stack_node = self.stack[-3]
        if stack_node.nonterminal == 'funcdef':
            context = get_user_context(self._module_context, self._position)
            node = search_ancestor(leaf, 'error_node', 'funcdef')
            if node is not None:
                if node.type == 'error_node':
                    n = node.children[0]
                    if n.type == 'decorators':
                        decorators = n.children
                    elif n.type == 'decorator':
                        decorators = [n]
                    else:
                        decorators = []
                else:
                    decorators = node.get_decorators()
                function_name = stack_node.nodes[1]
                return complete_param_names(context, function_name.value, decorators)
        return []

    def _complete_keywords(
        self,
        allowed_transitions: List[Union[str, PythonTokenTypes]],
        only_values: bool
    ) -> Generator[keywords.KeywordName, None, None]:
        for k in allowed_transitions:
            if isinstance(k, str) and k.isalpha():
                if not only_values or k in ('True', 'False', 'None'):
                    yield keywords.KeywordName(self._inference_state, k)

    def _complete_global_scope(self) -> List[AbstractNameDefinition]:
        context = get_user_context(self._module_context, self._position)
        debug.dbg('global completion scope: %s', context)
        flow_scope_node = get_flow_scope_node(self._module_node, self._position)
        filters = get_global_filters(context, self._position, flow_scope_node)
        completion_names = []
        for filter in filters:
            completion_names += filter.values()
        return completion_names

    def _complete_trailer(self, previous_leaf: Leaf) -> Tuple[Optional[str], List[AbstractNameDefinition]]:
        inferred_context = self._module_context.create_context(previous_leaf)
        values = infer_call_of_leaf(inferred_context, previous_leaf)
        debug.dbg('trailer completion values: %s', values, color='MAGENTA')
        cached_name = None
        if len(values) == 1:
            v, = values
            if v.is_module():
                if len(v.string_names) == 1:
                    module_name = v.string_names[0]
                    if module_name in ('numpy', 'tensorflow', 'matplotlib', 'pandas'):
                        cached_name = module_name
        return (cached_name, self._complete_trailer_for_values(values))

    def _complete_trailer_for_values(self, values: ValueSet) -> List[AbstractNameDefinition]:
        user_context = get_user_context(self._module_context, self._position)
        return complete_trailer(user_context, values)

    def _get_importer_names(
        self,
        names: List[Leaf],
        level: int = 0,
        only_modules: bool = True
    ) -> List[AbstractNameDefinition]:
        names = [n.value for n in names]
        i = imports.Importer(self._inference_state, names, self._module_context, level)
        return i.completion_names(self._inference_state, only_modules=only_modules)

    def _complete_inherited(self, is_function: bool = True) -> Generator[AbstractNameDefinition, None, None]:
        leaf = self._module_node.get_leaf_for_position(self._position, include_prefixes=True)
        cls = tree.search_ancestor(leaf, 'classdef')
        if cls is None:
            return
        class_value = self._module_context.create_value(cls)
        if cls.start_pos[1] >= leaf.start_pos[1]:
            return
        filters = class_value.get_filters(is_instance=True)
        next(filters)
        for filter in filters:
            for name in filter.values():
                if (name.api_type == 'function') == is_function:
                    yield name

    def _complete_in_string(self, start_leaf: Leaf, string: str) -> List[Completion]:
        def iter_relevant_lines(lines: List[str]) -> Generator[Optional[str], None, None]:
            include_next_line = False
            for l in code_lines:
                if include_next_line or l.startswith('>>>') or l.startswith(' '):
                    yield re.sub('^( *>>> ?| +)', '', l)
                else:
                    yield None
                include_next_line = bool(re.match(' *>>>', l))
        string = dedent(string)
        code_lines = split_lines(string, keepends=True)
        relevant_code_lines = list(iter_relevant_lines(code_lines))
        if relevant_code_lines[-1] is not None:
            relevant_code_lines = ['\n' if c is None else c for c in relevant_code_lines]
            return self._complete_code_lines(relevant_code_lines)
        match = re.search('`([^`\\s]+)', code_lines[-1])
        if match:
            return self._complete_code_lines([match.group(1)])
        return []

    def _complete_code_lines(self, code_lines: List[str]) -> List[Completion]:
        module_node = self._inference_state.grammar.parse(''.join(code_lines))
        module_value = DocstringModule(in_module_context=self._module_context, inference_state=self._inference_state, module_node=module_node, code_lines=code_lines)
        return Completion(self._inference_state, module_value.as_context(), code_lines=code_lines, position=module_node.end_pos, signatures_callback=lambda *args, **kwargs: [], fuzzy=self._fuzzy).complete()

def _gather_nodes(stack: List[Any]) -> List[Any]:
    nodes = []
    for stack_node in stack:
        if stack_node.dfa.from_rule == 'small_stmt':
            nodes = []
        else:
            nodes += stack_node.nodes
    return nodes

_string_start = re.compile('^\\w*(\\\'{3}|"{3}|\\\'|")')

def _extract_string_while_in_string(
    leaf: Leaf,
    position: Tuple[int, int]
) -> Tuple[Optional[str], Optional[Leaf], Optional[str]]:
    def return_part_of_leaf(leaf: Leaf) -> Tuple[Optional[str], Optional[Leaf], Optional[str]]:
        kwargs: Dict[str, Any] = {}
        if leaf.line == position[0]:
            kwargs['endpos'] = position[1] - leaf.column
        match = _string_start.match(leaf.value, **kwargs)
        if not match:
            return (None