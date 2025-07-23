"""
Like described in the :mod:`parso.python.tree` module,
there's a need for an ast like module to represent the states of parsed
modules.

But now there are also structures in Python that need a little bit more than
that. An ``Instance`` for example is only a ``Class`` before it is
instantiated. This class represents these cases.

So, why is there also a ``Class`` class here? Well, there are decorators and
they change classes in Python 3.

Representation modules also define "magic methods". Those methods look like
``py__foo__`` and are typically mappable to the Python equivalents ``__call__``
and others. Here's a list:

====================================== ========================================
**Method**                             **Description**
-------------------------------------- ----------------------------------------
py__call__(arguments: Array)           On callable objects, returns types.
py__bool__()                           Returns True/False/None; None means that
                                       there's no certainty.
py__bases__()                          Returns a list of base classes.
py__iter__()                           Returns a generator of a set of types.
py__class__()                          Returns the class of an instance.
py__simple_getitem__(index: int/str)   Returns a a set of types of the index.
                                       Can raise an IndexError/KeyError.
py__getitem__(indexes: ValueSet)       Returns a a set of types of the index.
py__file__()                           Only on modules. Returns None if does
                                       not exist.
py__package__() -> List[str]           Only on modules. For the import system.
py__path__()                           Only on modules. For the import system.
py__get__(call_object)                 Only on instances. Simulates
                                       descriptors.
py__doc__()                            Returns the docstring for a value.
====================================== ========================================
"""
from typing import Any, Dict, Iterator, List, Optional, Tuple

from jedi import debug
from jedi.inference.arguments import ValuesArguments, unpack_arglist
from jedi.inference.base_value import (
    NO_VALUES,
    ValueSet,
    iterator_to_value_set,
)
from jedi.inference.cache import (
    CachedMetaClass,
    inference_state_method_cache,
    inference_state_method_generator_cache,
)
from jedi.inference.compiled import builtin_from_name
from jedi.inference.context import ClassContext
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
from jedi.inference.lazy_value import LazyKnownValues, LazyTreeValue
from jedi.inference.names import TreeNameDefinition, ValueName
from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value.function import FunctionAndClassBase
from jedi.inference.gradual.base import GenericClass
from jedi.inference.gradual.typing import TypedDict, TypedDictClass
from jedi.inference.syntax_tree import expr_is_dotted
from jedi.inference import compiled
from jedi.inference.gradual.annotation import find_unknown_type_vars
from jedi.plugins import plugin_manager


class ClassName(TreeNameDefinition):

    def __init__(
        self,
        class_value: "ClassValue",
        tree_name: Any,
        name_context: Any,
        apply_decorators: bool,
    ) -> None:
        super().__init__(name_context, tree_name)
        self._apply_decorators = apply_decorators
        self._class_value = class_value

    @iterator_to_value_set
    def infer(self) -> Iterator["Value"]:
        inferred = tree_name_to_values(
            self.parent_context.inference_state, self.parent_context, self.tree_name
        )
        for result_value in inferred:
            if self._apply_decorators:
                yield from result_value.py__get__(instance=None, class_value=self._class_value)
            else:
                yield result_value

    @property
    def api_type(self) -> str:
        type_ = super().api_type
        if type_ == 'function':
            definition = self.tree_name.get_definition()
            if function_is_property(definition):
                return 'property'
        return type_


class ClassFilter(ParserTreeFilter):

    def __init__(
        self,
        class_value: "ClassValue",
        node_context: Optional[Any] = None,
        until_position: Optional[int] = None,
        origin_scope: Optional[Any] = None,
        is_instance: bool = False,
    ) -> None:
        super().__init__(
            class_value.as_context(),
            node_context,
            until_position=until_position,
            origin_scope=origin_scope,
        )
        self._class_value = class_value
        self._is_instance = is_instance

    def _convert_names(self, names: List[Any]) -> List[ClassName]:
        return [
            ClassName(
                class_value=self._class_value,
                tree_name=name,
                name_context=self._node_context,
                apply_decorators=not self._is_instance,
            )
            for name in names
        ]

    def _equals_origin_scope(self) -> bool:
        node = self._origin_scope
        while node is not None:
            if node == self._parser_scope or node == self.parent_context:
                return True
            node = get_cached_parent_scope(self._parso_cache_node, node)
        return False

    def _access_possible(self, name: Any) -> bool:
        if not self._is_instance:
            expr_stmt = name.get_definition()
            if expr_stmt is not None and expr_stmt.type == 'expr_stmt':
                annassign = expr_stmt.children[1]
                if annassign.type == 'annassign':
                    if 'ClassVar' not in annassign.children[1].get_code() and '=' not in annassign.children:
                        return False
        return (
            not name.value.startswith('__')
            or name.value.endswith('__')
            or self._equals_origin_scope()
        )

    def _filter(self, names: List[Any]) -> List[ClassName]:
        names = super()._filter(names)
        return [name for name in names if self._access_possible(name)]


class ClassMixin:

    def is_class(self) -> bool:
        return True

    def is_class_mixin(self) -> bool:
        return True

    def py__call__(self, arguments: ValuesArguments) -> ValueSet:
        if self.is_typeddict():
            return ValueSet([TypedDict(self)])
        return ValueSet(
            [TreeInstance(self.inference_state, self.parent_context, self, arguments)]
        )

    def py__class__(self) -> ValueSet:
        return builtin_from_name(self.inference_state, 'type')

    @property
    def name(self) -> ValueName:
        return ValueName(self, self.tree_node.name)

    def py__name__(self) -> str:
        return self.name.string_name

    @inference_state_method_generator_cache()
    def py__mro__(self) -> Iterator["Value"]:
        mro: List["Value"] = [self]
        yield self
        for lazy_cls in self.py__bases__():
            for cls in lazy_cls.infer():
                try:
                    mro_method = cls.py__mro__
                except AttributeError:
                    debug.warning('Super class of %s is not a class: %s', self, cls)
                else:
                    for cls_new in mro_method():
                        if cls_new not in mro:
                            mro.append(cls_new)
                            yield cls_new

    def get_filters(
        self,
        origin_scope: Optional[Any] = None,
        is_instance: bool = False,
        include_metaclasses: bool = True,
        include_type_when_class: bool = True,
    ) -> Iterator[Any]:
        if include_metaclasses:
            metaclasses = self.get_metaclasses()
            if metaclasses:
                yield from self.get_metaclass_filters(metaclasses, is_instance)
        for cls in self.py__mro__():
            if cls.is_compiled():
                yield from cls.get_filters(is_instance=is_instance)
            else:
                yield ClassFilter(
                    self,
                    node_context=cls.as_context(),
                    origin_scope=origin_scope,
                    is_instance=is_instance,
                )
        if not is_instance and include_type_when_class:
            type_ = builtin_from_name(self.inference_state, 'type')
            assert isinstance(type_, ClassValue)
            if type_ != self:
                args = ValuesArguments([])
                for instance in type_.py__call__(args):
                    instance_filters = instance.get_filters()
                    next(instance_filters, None)
                    next(instance_filters, None)
                    x = next(instance_filters, None)
                    assert x is not None
                    yield x

    def get_signatures(self) -> List[Any]:
        metaclasses = self.get_metaclasses()
        if metaclasses:
            sigs = self.get_metaclass_signatures(metaclasses)
            if sigs:
                return sigs
        args = ValuesArguments([])
        init_funcs = self.py__call__(args).py__getattribute__('__init__')
        return [sig.bind(self) for sig in init_funcs.get_signatures()]

    def _as_context(self) -> ClassContext:
        return ClassContext(self)

    def get_type_hint(self, add_class_info: bool = True) -> str:
        if add_class_info:
            return f'Type[{self.py__name__()}]'
        return self.py__name__()

    @inference_state_method_cache(default=False)
    def is_typeddict(self) -> bool:
        for lazy_cls in self.py__bases__():
            if not isinstance(lazy_cls, LazyTreeValue):
                return False
            tree_node = lazy_cls.data
            if not expr_is_dotted(tree_node):
                return False
            for cls in lazy_cls.infer():
                if isinstance(cls, TypedDictClass):
                    return True
                try:
                    method = cls.is_typeddict
                except AttributeError:
                    return False
                else:
                    if method():
                        return True
        return False

    def py__getitem__(
        self, index_value_set: ValueSet, contextualized_node: Any
    ) -> ValueSet:
        if not index_value_set:
            debug.warning('Class indexes inferred to nothing. Returning class instead')
            return ValueSet([self])
        return ValueSet(
            (
                GenericClass(
                    self,
                    LazyGenericManager(
                        context_of_index=contextualized_node.context,
                        index_value=index_value,
                    ),
                )
                for index_value in index_value_set
            )
        )

    def with_generics(self, generics_tuple: Tuple[Any, ...]) -> ValueSet:
        return ValueSet(
            [GenericClass(self, TupleGenericManager(generics_tuple))]
        )

    def define_generics(self, type_var_dict: Dict[str, Any]) -> ValueSet:
        def remap_type_vars() -> Iterator[Any]:
            """
            The TypeVars in the resulting classes have sometimes different names
            and we need to check for that, e.g. a signature can be:

            def iter(iterable: Iterable[_T]) -> Iterator[_T]: ...

            However, the iterator is defined as Iterator[_T_co], which means it has
            a different type var name.
            """
            for type_var in self.list_type_vars():
                yield type_var_dict.get(type_var.py__name__(), NO_VALUES)

        if type_var_dict:
            return ValueSet(
                [
                    GenericClass(
                        self, TupleGenericManager(tuple(remap_type_vars()))
                    )
                ]
            )
        return ValueSet({self})


class ClassValue(ClassMixin, FunctionAndClassBase, metaclass=CachedMetaClass):
    api_type: str = 'class'

    @inference_state_method_cache()
    def list_type_vars(self) -> List[Any]:
        found: List[Any] = []
        arglist = self.tree_node.get_super_arglist()
        if arglist is None:
            return []
        for stars, node in unpack_arglist(arglist):
            if stars:
                continue
            for type_var in find_unknown_type_vars(self.parent_context, node):
                if type_var not in found:
                    found.append(type_var)
        return found

    def _get_bases_arguments(self) -> Optional[Any]:
        arglist = self.tree_node.get_super_arglist()
        if arglist:
            from jedi.inference import arguments
            return arguments.TreeArguments(self.inference_state, self.parent_context, arglist)
        return None

    @inference_state_method_cache(default=())
    def py__bases__(self) -> ValueSet:
        args = self._get_bases_arguments()
        if args is not None:
            lst = [value for key, value in args.unpack() if key is None]
            if lst:
                return lst
        if self.py__name__() == 'object' and self.parent_context.is_builtins_module():
            return ValueSet([])
        return ValueSet([
            LazyKnownValues(
                self.inference_state.builtins_module.py__getattribute__('object')
            )
        ])

    @plugin_manager.decorate()
    def get_metaclass_filters(
        self, metaclasses: ValueSet, is_instance: bool
    ) -> List[Any]:
        debug.warning('Unprocessed metaclass %s', metaclasses)
        return []

    @inference_state_method_cache(default=NO_VALUES)
    def get_metaclasses(self) -> ValueSet:
        args = self._get_bases_arguments()
        if args is not None:
            m = [value for key, value in args.unpack() if key == 'metaclass']
            metaclasses = ValueSet.from_sets((lazy_value.infer() for lazy_value in m))
            metaclasses = ValueSet((m for m in metaclasses if m.is_class()))
            if metaclasses:
                return metaclasses
        for lazy_base in self.py__bases__():
            for value in lazy_base.infer():
                if value.is_class():
                    values = value.get_metaclasses()
                    if values:
                        return values
        return NO_VALUES

    @plugin_manager.decorate()
    def get_metaclass_signatures(self, metaclasses: ValueSet) -> List[Any]:
        return []
