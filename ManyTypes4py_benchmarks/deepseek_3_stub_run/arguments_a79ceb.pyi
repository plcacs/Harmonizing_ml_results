import re
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload
)
from parso.python import tree
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import (
    LazyKnownValue,
    LazyKnownValues,
    LazyTreeValue,
    LazyValue,
    MergedLazyValue
)
from jedi.inference.names import (
    ParamName,
    TreeNameDefinition,
    AnonymousParamName,
    AbstractNameDefinition
)
from jedi.inference.base_value import (
    NO_VALUES,
    ValueSet,
    ContextualizedNode,
    BaseValue,
    Value
)
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.inference import InferenceState

_T = TypeVar("_T")
_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])
_ValueT = TypeVar("_ValueT", bound=Value)

def try_iter_content(types: ValueSet, depth: int = 0) -> None:
    ...

class ParamIssue(Exception):
    ...

def repack_with_argument_clinic(
    clinic_string: str
) -> Callable[[_FuncT], Callable[[Value, "AbstractArguments"], ValueSet]]:
    ...

def iterate_argument_clinic(
    inference_state: InferenceState,
    arguments: "AbstractArguments",
    clinic_string: str
) -> Generator[ValueSet, None, None]:
    ...

def _parse_argument_clinic(string: str) -> Generator[Tuple[str, bool, bool, int], None, None]:
    ...

class _AbstractArgumentsMixin:
    def unpack(
        self, funcdef: Optional[tree.Function] = None
    ) -> Generator[Tuple[Optional[str], LazyValue], None, None]:
        ...

    def get_calling_nodes(self) -> List[ContextualizedNode]:
        ...

class AbstractArguments(_AbstractArgumentsMixin):
    context: Optional[Any] = ...
    argument_node: Optional[tree.BaseNode] = ...
    trailer: Optional[tree.BaseNode] = ...

def unpack_arglist(
    arglist: Optional[tree.BaseNode]
) -> Generator[Tuple[int, tree.BaseNode], None, None]:
    ...

class TreeArguments(AbstractArguments):
    def __init__(
        self,
        inference_state: InferenceState,
        context: Any,
        argument_node: Union[tree.BaseNode, List[tree.BaseNode]],
        trailer: Optional[tree.BaseNode] = None
    ) -> None:
        ...

    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(cls: Type[_T], *args: Any, **kwargs: Any) -> _T:
        ...

    def unpack(
        self, funcdef: Optional[tree.Function] = None
    ) -> Generator[Tuple[Optional[str], LazyValue], None, None]:
        ...

    def _as_tree_tuple_objects(
        self
    ) -> Generator[Tuple[tree.BaseNode, Optional[tree.BaseNode], int], None, None]:
        ...

    def iter_calling_names_with_star(
        self
    ) -> Generator[TreeNameDefinition, None, None]:
        ...

    def __repr__(self) -> str:
        ...

    def get_calling_nodes(self) -> List[ContextualizedNode]:
        ...

class ValuesArguments(AbstractArguments):
    def __init__(self, values_list: List[ValueSet]) -> None:
        ...

    def unpack(
        self, funcdef: Optional[tree.Function] = None
    ) -> Generator[Tuple[Optional[str], LazyValue], None, None]:
        ...

    def __repr__(self) -> str:
        ...

class TreeArgumentsWrapper(_AbstractArgumentsMixin):
    def __init__(self, arguments: TreeArguments) -> None:
        ...

    @property
    def context(self) -> Any:
        ...

    @property
    def argument_node(self) -> Optional[tree.BaseNode]:
        ...

    @property
    def trailer(self) -> Optional[tree.BaseNode]:
        ...

    def unpack(
        self, func: Optional[tree.Function] = None
    ) -> Generator[Tuple[Optional[str], LazyValue], None, None]:
        ...

    def get_calling_nodes(self) -> List[ContextualizedNode]:
        ...

    def __repr__(self) -> str:
        ...

def _iterate_star_args(
    context: Any,
    array: Value,
    input_node: tree.BaseNode,
    funcdef: Optional[tree.Function] = None
) -> Generator[LazyValue, None, None]:
    ...

def _star_star_dict(
    context: Any,
    array: Value,
    input_node: tree.BaseNode,
    funcdef: Optional[tree.Function]
) -> Dict[str, LazyValue]:
    ...