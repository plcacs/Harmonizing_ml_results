from collections import defaultdict
from inspect import Parameter
from typing import Any, Dict, List, Optional, Tuple, Union
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyTreeValue, LazyUnknownValue
from jedi.inference.value import iterable
from jedi.inference.names import ParamName
from jedi.inference.base_value import Value
from jedi.inference.arguments import AbstractArguments
from jedi.inference.context import Context
from jedi.inference.value.function import FunctionValue
from jedi.inference.gradual.base import DefineGenericBase
from parso.python.tree import Param, Name
from parso.tree import BaseNode

def _add_argument_issue(error_name: str, lazy_value: LazyTreeValue, message: str) -> Optional[analysis.Issue]:
    if isinstance(lazy_value, LazyTreeValue):
        node: BaseNode = lazy_value.data
        if node.parent.type == 'argument':
            node = node.parent
        return analysis.add(lazy_value.context, error_name, node, message)
    return None

class ExecutedParamName(ParamName):
    def __init__(
        self,
        function_value: FunctionValue,
        arguments: AbstractArguments,
        param_node: Param,
        lazy_value: Union[LazyKnownValue, LazyTreeValue, LazyUnknownValue],
        is_default: bool = False
    ) -> None:
        super().__init__(function_value, param_node.name, arguments=arguments)
        self._lazy_value: Union[LazyKnownValue, LazyTreeValue, LazyUnknownValue] = lazy_value
        self._is_default: bool = is_default

    def infer(self) -> Value:
        return self._lazy_value.infer()

    def matches_signature(self) -> bool:
        if self._is_default:
            return True
        argument_values: Value = self.infer().py__class__()
        if self.get_kind() in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            return True
        annotations: Optional[DefineGenericBase] = self.infer_annotation(execute_annotation=False)
        if not annotations:
            return True
        matches: bool = any((c1.is_sub_class_of(c2) for c1 in argument_values for c2 in annotations.gather_annotation_classes()))
        debug.dbg('param compare %s: %s <=> %s', matches, argument_values, annotations, color='BLUE')
        return matches

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.string_name)

def get_executed_param_names_and_issues(
    function_value: FunctionValue,
    arguments: AbstractArguments
) -> Tuple[List[ExecutedParamName], List[Optional[analysis.Issue]]]:
    def too_many_args(argument: Union[LazyKnownValue, LazyTreeValue]) -> None:
        m: str = _error_argument_count(funcdef, len(unpacked_va))
        if arguments.get_calling_nodes():
            issues.append(_add_argument_issue('type-error-too-many-arguments', argument, message=m))
        else:
            issues.append(None)
            debug.warning('non-public warning: %s', m)

    issues: List[Optional[analysis.Issue]] = []
    result_params: List[ExecutedParamName] = []
    param_dict: Dict[str, Param] = {}
    funcdef: FunctionValue.tree_node = function_value.tree_node
    default_param_context: Context = function_value.get_default_param_context()
    for param in funcdef.get_params():
        param_dict[param.name.value] = param
    unpacked_va: List[Any] = list(arguments.unpack(funcdef))
    var_arg_iterator: PushBackIterator = PushBackIterator(iter(unpacked_va))
    non_matching_keys: Dict[str, Any] = defaultdict(lambda: [])
    keys_used: Dict[str, Union[ExecutedParamName, LazyKnownValue, LazyTreeValue, LazyUnknownValue]] = {}
    keys_only: bool = False
    had_multiple_value_error: bool = False
    for param in funcdef.get_params():
        is_default: bool = False
        key: Optional[str]
        argument: Optional[Union[LazyKnownValue, LazyTreeValue, LazyUnknownValue]]
        key, argument = next(var_arg_iterator, (None, None))
        while key is not None:
            keys_only = True
            try:
                key_param: Param = param_dict[key]
            except KeyError:
                non_matching_keys[key] = argument
            else:
                if key in keys_used:
                    had_multiple_value_error = True
                    m: str = "TypeError: %s() got multiple values for keyword argument '%s'." % (funcdef.name, key)
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
            lazy_value_list: List[Union[LazyKnownValue, LazyTreeValue, LazyUnknownValue]] = []
            if argument is not None:
                lazy_value_list.append(argument)
                for key, argument in var_arg_iterator:
                    if key:
                        var_arg_iterator.push_back((key, argument))
                        break
                    lazy_value_list.append(argument)
            seq: iterable.FakeTuple = iterable.FakeTuple(function_value.inference_state, lazy_value_list)
            result_arg: LazyKnownValue = LazyKnownValue(seq)
        elif param.star_count == 2:
            if argument is not None:
                too_many_args(argument)
            dct: iterable.FakeDict = iterable.FakeDict(function_value.inference_state, dict(non_matching_keys))
            result_arg: LazyKnownValue = LazyKnownValue(dct)
            non_matching_keys = {}
        elif argument is None:
            if param.default is None:
                result_arg: LazyUnknownValue = LazyUnknownValue()
                if not keys_only:
                    for contextualized_node in arguments.get_calling_nodes():
                        m: str = _error_argument_count(funcdef, len(unpacked_va))
                        issues.append(analysis.add(contextualized_node.context, 'type-error-too-few-arguments', contextualized_node.node, message=m))
            else:
                result_arg: LazyTreeValue = LazyTreeValue(default_param_context, param.default)
                is_default = True
        else:
            result_arg: Union[LazyKnownValue, LazyTreeValue, LazyUnknownValue] = argument
        result_params.append(ExecutedParamName(function_value, arguments, param, result_arg, is_default=is_default))
        if not isinstance(result_arg, LazyUnknownValue):
            keys_used[param.name.value] = result_params[-1]
    if keys_only:
        for k in set(param_dict) - set(keys_used):
            param: Param = param_dict[k]
            if not (non_matching_keys or had_multiple_value_error or param.star_count or param.default):
                for contextualized_node in arguments.get_calling_nodes():
                    m: str = _error_argument_count(funcdef, len(unpacked_va))
                    issues.append(analysis.add(contextualized_node.context, 'type-error-too-few-arguments', contextualized_node.node, message=m))
    for key, lazy_value in non_matching_keys.items():
        m: str = "TypeError: %s() got an unexpected keyword argument '%s'." % (funcdef.name, key)
        issues.append(_add_argument_issue('type-error-keyword-argument', lazy_value, message=m))
    remaining_arguments: List[Tuple[Optional[str], Union[LazyKnownValue, LazyTreeValue, LazyUnknownValue]]] = list(var_arg_iterator)
    if remaining_arguments:
        first_key: Optional[str]
        lazy_value: Union[LazyKnownValue, LazyTreeValue, LazyUnknownValue]
        first_key, lazy_value = remaining_arguments[0]
        too_many_args(lazy_value)
    return (result_params, issues)

def get_executed_param_names(
    function_value: FunctionValue,
    arguments: AbstractArguments
) -> List[ExecutedParamName]:
    return get_executed_param_names_and_issues(function_value, arguments)[0]

def _error_argument_count(funcdef: FunctionValue.tree_node, actual_count: int) -> str:
    params: List[Param] = funcdef.get_params()
    default_arguments: int = sum((1 for p in params if p.default or p.star_count))
    if default_arguments == 0:
        before: str = 'exactly '
    else:
        before: str = 'from %s to ' % (len(params) - default_arguments)
    return 'TypeError: %s() takes %s%s arguments (%s given).' % (funcdef.name, before, len(params), actual_count)
