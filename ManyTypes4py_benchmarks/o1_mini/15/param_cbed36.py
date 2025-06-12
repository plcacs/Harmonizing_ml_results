from collections import defaultdict
from inspect import Parameter
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from jedi import debug
from jedi.inference import analysis
from jedi.inference.arguments import Arguments
from jedi.inference.context import Context
from jedi.inference.lazy_value import (
    LazyKnownValue,
    LazyTreeValue,
    LazyUnknownValue,
)
from jedi.inference.names import ParamName
from jedi.inference.tree import TreeNode
from jedi.inference.types import FunctionValue
from jedi.inference.value import IterableValue
from jedi.inference.utils import PushBackIterator


def _add_argument_issue(
    error_name: str, lazy_value: LazyTreeValue, message: str
) -> Optional[analysis.Issue]:
    if isinstance(lazy_value, LazyTreeValue):
        node = lazy_value.data
        if node.parent.type == 'argument':
            node = node.parent
        return analysis.add(lazy_value.context, error_name, node, message)
    return None


class ExecutedParamName(ParamName):
    def __init__(
        self,
        function_value: FunctionValue,
        arguments: Arguments,
        param_node: TreeNode,
        lazy_value: Union[LazyKnownValue, LazyTreeValue, LazyUnknownValue],
        is_default: bool = False,
    ) -> None:
        super().__init__(function_value, param_node.name.value, arguments=arguments)
        self._lazy_value = lazy_value
        self._is_default = is_default

    def infer(self) -> List[IterableValue]:
        return self._lazy_value.infer()

    def matches_signature(self) -> bool:
        if self._is_default:
            return True
        argument_values = self.infer().py__class__()
        if self.get_kind() in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            return True
        annotations = self.infer_annotation(execute_annotation=False)
        if not annotations:
            return True
        matches = any(
            c1.is_sub_class_of(c2)
            for c1 in argument_values
            for c2 in annotations.gather_annotation_classes()
        )
        debug.dbg(
            'param compare %s: %s <=> %s',
            matches,
            argument_values,
            annotations,
            color='BLUE',
        )
        return matches

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.string_name}>'


def get_executed_param_names_and_issues(
    function_value: FunctionValue, arguments: Arguments
) -> Tuple[List[ExecutedParamName], List[Optional[analysis.Issue]]]:
    """
    Return a tuple of:
      - a list of `ExecutedParamName`s corresponding to the arguments of the
        function execution `function_value`, containing the inferred value of
        those arguments (whether explicit or default)
      - a list of the issues encountered while building that list

    For example, given:
    