from collections import defaultdict
from inspect import Parameter
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyTreeValue, LazyUnknownValue
from jedi.inference.value import iterable
from jedi.inference.names import ParamName
from typing import Any, DefaultDict, List, Optional, Tuple, Union

def _add_argument_issue(error_name: str, lazy_value: Union[LazyTreeValue, LazyKnownValue, LazyUnknownValue], message: str) -> Optional[Any]:
    if isinstance(lazy_value, LazyTreeValue):
        node = lazy_value.data
        if node.parent.type == 'argument':
            node = node.parent
        return analysis.add(lazy_value.context, error_name, node, message)
    return None

class ExecutedParamName(ParamName):
    def __init__(self, function_value: Any, arguments: Any, param_node: Any, lazy_value: Any, is_default: bool = False) -> None:
        super().__init__(function_value, param_node.name, arguments=arguments)
        self._lazy_value = lazy_value
        self._is_default = is_default

    def infer(self) -> Any:
        return self._lazy_value.infer()

    def matches_signature(self) -> bool:
        if self._is_default:
            return True
        argument_values = self.infer().py__class__()
        if self.get_kind() in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            return True
        annotations = self.infer_annotation(execute_annotation=False)
        if not annotations:
            return True
        matches = any((c1.is_sub_class_of(c2) for c1 in argument_values for c2 in annotations.gather_annotation_classes()))
        debug.dbg('param compare %s: %s <=> %s', matches, argument_values, annotations, color='BLUE')
        return matches

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.string_name)

def get_executed_param_names_and_issues(function_value: Any, arguments: Any) -> Tuple[List[ExecutedParamName], List[Optional[Any]]]:
    """
    Return a tuple of:
      - a list of `ExecutedParamName`s corresponding to the arguments of the
        function execution `function_value`, containing the inferred value of
        those arguments (whether explicit or default)
      - a list of the issues encountered while building that list

    For example, given:
    ```
    def foo(a, b, c=None, d='d'): ...

    foo(42, c='c')
    ```

    Then for the execution of `foo`, this will return a tuple containing:
      - a list with entries for each parameter a, b, c & d; the entries for a,
        c, & d will have their values (42, 'c' and 'd' respectively) included.
      - a list with a single entry about the lack of a value for `b`
    """
    def too_many_args(argument: Any) -> None:
        m = _error_argument_count(funcdef, len(unpacked_va))
        if arguments.get_calling_nodes():
            issues.append(_add_argument_issue('type-error-too-many-arguments', argument, message=m))
        else:
            issues.append(None)
            debug.warning('non-public warning: %s', m)
    issues: List[Optional[Any]] = []
    result_params: List[ExecutedParamName] = []
    param_dict: Dict[str, Any] = {}
    funcdef: Any = function_value.tree_node
    default_param_context: Any = function_value.get_default_param_context()
    for param in funcdef.get_params():
        param_dict[param.name.value] = param
    unpacked_va: List[Tuple[Optional[str], Any]] = list(arguments.unpack(funcdef))
    var_arg_iterator: PushBackIterator[Tuple[Optional[str], Any]] = PushBackIterator(iter(unpacked_va))
    non_matching_keys: DefaultDict[str, List[Any]] = defaultdict(lambda: [])
    keys_used: Dict[str, ExecutedParamName] = {}
    keys_only: bool = False
    had_multiple_value_error: bool = False
    for param in funcdef.get_params():
        is_default: bool = False
        key, argument = next(var_arg_iterator, (None, None))
        while key is not None:
            keys_only = True
            try:
                key_param = param_dict[key]
            except KeyError:
                non_matching_keys[key].append(argument)
            else:
                if key in keys_used:
                    had_multiple_value_error = True
                    m = "TypeError: %s() got multiple values for keyword argument '%s'." % (funcdef.name, key)
                    for contextualized_node in arguments.get_calling_nodes():
                        issues.append(analysis.add(contextualized_node.context, 'type-error-multiple-values', contextualized_node.node, message=m))
                else:
                    keys_used[key] = ExecutedParamName(function_value, arguments, key_param, argument)
            key, argument = next(var_arg_iterator, (None, None))
        try:
            result_params.append(keys_used[param.name.value])
            continue
        except KeyError:
            pass
        if param.star_count == 1:
            lazy_value_list: List[Any] = []
            if argument is not None:
                lazy_value_list.append(argument)
                for key, argument in var_arg_iterator:
                    if key:
                        var_arg_iterator.push_back((key, argument))
                        break
                    lazy_value_list.append(argument)
            seq = iterable.FakeTuple(function_value.inference_state, lazy_value_list)
            result_arg = LazyKnownValue(seq)
        elif param.star_count == 2:
            if argument is not None:
                too_many_args(argument)
            dct = iterable.FakeDict(function_value.inference_state, dict(non_matching_keys))
            result_arg = LazyKnownValue(dct)
            non_matching_keys = {}
        elif argument is None:
            if param.default is None:
                result_arg = LazyUnknownValue()
                if not keys_only:
                    for contextualized_node in arguments.get_calling_nodes():
                        m = _error_argument_count(funcdef, len(unpacked_va))
                        issues.append(analysis.add(contextualized_node.context, 'type-error-too-few-arguments', contextualized_node.node, message=m))
            else:
                result_arg = LazyTreeValue(default_param_context, param.default)
                is_default = True
        else:
            result_arg = argument
        result_params.append(ExecutedParamName(function_value, arguments, param, result_arg, is_default=is_default))
        if not isinstance(result_arg, LazyUnknownValue):
            keys_used[param.name.value] = result_params[-1]
    if keys_only:
        for k in set(param_dict) - set(keys_used):
            param = param_dict[k]
            if not (non_matching_keys or had_multiple_value_error or param.star_count or param.default):
                for contextualized_node in arguments.get_calling_nodes():
                    m = _error_argument_count(funcdef, len(unpacked_va))
                    issues.append(analysis.add(contextualized_node.context, 'type-error-too-few-arguments', contextualized_node.node, message=m))
    for key, lazy_values in non_matching_keys.items():
        # Report for each key the first encountered lazy value
        if lazy_values:
            m = "TypeError: %s() got an unexpected keyword argument '%s'." % (funcdef.name, key)
            issues.append(_add_argument_issue('type-error-keyword-argument', lazy_values[0], message=m))
    remaining_arguments: List[Tuple[Optional[str], Any]] = list(var_arg_iterator)
    if remaining_arguments:
        first_key, lazy_value = remaining_arguments[0]
        too_many_args(lazy_value)
    return (result_params, issues)

def get_executed_param_names(function_value: Any, arguments: Any) -> List[ExecutedParamName]:
    """
    Return a list of `ExecutedParamName`s corresponding to the arguments of the
    function execution `function_value`, containing the inferred value of those
    arguments (whether explicit or default). Any issues building this list (for
    example required arguments which are missing in the invocation) are ignored.

    For example, given:
    ```
    def foo(a, b, c=None, d='d'): ...

    foo(42, c='c')
    ```

    Then for the execution of `foo`, this will return a list containing entries
    for each parameter a, b, c & d; the entries for a, c, & d will have their
    values (42, 'c' and 'd' respectively) included.
    """
    return get_executed_param_names_and_issues(function_value, arguments)[0]

def _error_argument_count(funcdef: Any, actual_count: int) -> str:
    params = funcdef.get_params()
    default_arguments = sum((1 for p in params if p.default or p.star_count))
    if default_arguments == 0:
        before = 'exactly '
    else:
        before = 'from %s to ' % (len(params) - default_arguments)
    return 'TypeError: %s() takes %s%s arguments (%s given).' % (funcdef.name, before, len(params), actual_count)