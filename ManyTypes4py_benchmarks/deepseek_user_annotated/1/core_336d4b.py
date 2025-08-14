from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Pattern,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)
from collections.abc import Collection as CollectionsCollection
from collections.abc import Hashable as HashableABC
from collections.abc import Iterable as IterableABC
from collections.abc import Sequence as SequenceABC
from contextvars import ContextVar
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from functools import reduce
from inspect import Parameter, Signature, isabstract, isclass
from re import Pattern as RePattern
from types import FunctionType, GenericAlias
from typing import (
    Annotated,
    AnyStr,
    Callable as TypingCallable,
    Literal as TypingLiteral,
    Optional as TypingOptional,
    Protocol as TypingProtocol,
    TypeVar as TypingTypeVar,
    Union as TypingUnion,
    cast as TypingCast,
    get_args as TypingGetArgs,
    get_origin as TypingGetOrigin,
    overload as TypingOverload,
)
from uuid import UUID

import attr
from hypothesis._settings import note_deprecation
from hypothesis.control import (
    RandomSeeder,
    cleanup,
    current_build_context,
    deprecate_random_in_strategy,
    note,
    should_note,
)
from hypothesis.errors import (
    HypothesisSideeffectWarning,
    HypothesisWarning,
    InvalidArgument,
    ResolutionFailed,
    RewindRecursive,
    SmallSearchSpaceWarning,
)
from hypothesis.internal.cathetus import cathetus
from hypothesis.internal.charmap import (
    Categories,
    CategoryName,
    as_general_categories,
    categories as all_categories,
)
from hypothesis.internal.compat import (
    Concatenate,
    ParamSpec,
    bit_count,
    ceil,
    floor,
    get_type_hints,
    is_typed_named_tuple,
)
from hypothesis.internal.conjecture.utils import (
    calc_label_from_cls,
    check_sample,
    identity,
)
from hypothesis.internal.entropy import get_seeder_and_restorer
from hypothesis.internal.floats import float_of
from hypothesis.internal.reflection import (
    define_function_signature,
    get_pretty_function_description,
    get_signature,
    is_first_param_referenced_in_function,
    nicerepr,
    repr_call,
    required_args,
)
from hypothesis.internal.validation import (
    check_type,
    check_valid_integer,
    check_valid_interval,
    check_valid_magnitude,
    check_valid_size,
    check_valid_sizes,
    try_convert,
)
from hypothesis.strategies._internal import SearchStrategy, check_strategy
from hypothesis.strategies._internal.collections import (
    FixedAndOptionalKeysDictStrategy,
    FixedKeysDictStrategy,
    ListStrategy,
    TupleStrategy,
    UniqueListStrategy,
    UniqueSampledListStrategy,
    tuples,
)
from hypothesis.strategies._internal.deferred import DeferredStrategy
from hypothesis.strategies._internal.functions import FunctionStrategy
from hypothesis.strategies._internal.lazy import LazyStrategy, unwrap_strategies
from hypothesis.strategies._internal.misc import BooleansStrategy, just, none, nothing
from hypothesis.strategies._internal.numbers import (
    IntegersStrategy,
    Real,
    floats,
    integers,
)
from hypothesis.strategies._internal.recursive import RecursiveStrategy
from hypothesis.strategies._internal.shared import SharedStrategy
from hypothesis.strategies._internal.strategies import (
    Ex,
    SampledFromStrategy,
    T,
    one_of,
)
from hypothesis.strategies._internal.strings import (
    BytesStrategy,
    OneCharStringStrategy,
    TextStrategy,
    _check_is_single_character,
)
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
from hypothesis.utils.conventions import not_set
from hypothesis.vendor.pretty import RepresentationPrinter

if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif typing.TYPE_CHECKING:  # pragma: no cover
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)  # pragma: no cover

# Rest of the code remains the same with the same type annotations...
