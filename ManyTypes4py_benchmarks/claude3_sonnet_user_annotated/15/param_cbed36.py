from collections import defaultdict
from inspect import Parameter
from typing import Dict, Iterator, List, Optional, Tuple, Union, Any, cast

from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, \
    LazyTreeValue, LazyUnknownValue, LazyValue
from jedi.inference.value import iterable
from jedi.inference.names import ParamName
from jedi.inference.base_value import ValueSet
from jedi.inference.value.function import FunctionValue
from jedi.inference.arguments import Arguments
from jedi.parser_utils import NodeOrLeaf


def _add_argument_issue(error_name: str, lazy_value: LazyValue, message: str) -> Optional[analysis.Error]:
    if isinstance(lazy_value, LazyTreeValue):
        node = lazy_value.data
        if node.parent.type == 'argument':
            node = node.parent
        return analysis.add(lazy_value.context, error_name, node, message)
    return None


class ExecutedParamName(ParamName):
    def __init__(self, function_value: FunctionValue, arguments: Arguments, param_node: NodeOrLeaf, lazy_value: LazyValue, is_default: bool = False) -> None:
        super().__init__(function_value, param_node.name, arguments=arguments)
        self._lazy_value = lazy_value
        self._is_default = is_default

    def infer(self) -> ValueSet:
        return self._lazy_value.infer()

    def matches_signature(self) -> bool:
        if self._is_default:
            return True
        argument_values = self.infer().py__class__()
        if self.get_kind() in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            return True
        annotations = self.infer_annotation(execute_annotation=False)
        if not annotations:
            # If we cannot infer annotations - or there aren't any - pretend
            # that the signature matches.
            return True
        matches = any(c1.is_sub_class_of(c2)
                      for c1 in argument_values
                      for c2 in annotations.gather_annotation_classes())
        debug.dbg("param compare %s: %s <=> %s",
                  matches, argument_values, annotations, color='BLUE')
        return matches

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.string_name)


def get_executed_param_names_and_issues(function_value: FunctionValue, arguments: Arguments) -> Tuple[List[ExecutedParamName], List[Optional[analysis.Error]]]:
    """
    Return a tuple of:
      - a list of `ExecutedParamName`s corresponding to the arguments of the
        function execution `function_value`, containing the inferred value of
        those arguments (whether explicit or default)
      - a list of the issues encountered while building that list

    For example, given:
    