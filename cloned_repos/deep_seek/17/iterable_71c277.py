"""
Contains all classes and functions to deal with lists, dicts, generators and
iterators in general.
"""
from typing import (
    Any, Dict, Generator, Iterator, List, Optional, Set, Tuple, Type, TypeVar, Union
)
from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, LazyTreeValue
from jedi.inference.helpers import get_int_or_none, is_string, reraise_getitem_errors, SimpleGetItemNotFound
from jedi.inference.utils import safe_property, to_list
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.filters import LazyAttributeOverwrite, publish_method
from jedi.inference.base_value import ValueSet, Value, NO_VALUES, ContextualizedNode, iterate_values, sentinel, LazyValueWrapper
from jedi.parser_utils import get_sync_comp_fors
from jedi.inference.context import CompForContext, Context
from jedi.inference.value.dynamic_arrays import check_array_additions
from jedi.inference.gradual.base import GenericClass
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference import InferenceState
from jedi.inference.names import AbstractNameDefinition
from jedi.inference.arguments import Arguments
from jedi.inference.value.iterable import FakeTuple, FakeList, FakeDict
from jedi.inference.value import Slice

T = TypeVar('T')

class IterableMixin:

    def py__next__(self, contextualized_node: Optional[ContextualizedNode] = None) -> ValueSet:
        return self.py__iter__(contextualized_node)

    def py__stop_iteration_returns(self) -> ValueSet:
        return ValueSet([compiled.builtin_from_name(self.inference_state, 'None')])
    
    def get_safe_value(self, default: Any = sentinel) -> Any:
        return Value.get_safe_value(self, default)

class GeneratorBase(LazyAttributeOverwrite, IterableMixin):
    array_type: Optional[str] = None

    def _get_wrapped_value(self) -> Value:
        instance, = self._get_cls().execute_annotation()
        return instance

    def _get_cls(self) -> Value:
        generator, = self.inference_state.typing_module.py__getattribute__('Generator')
        return generator

    def py__bool__(self) -> bool:
        return True

    @publish_method('__iter__')
    def _iter(self, arguments: Arguments) -> ValueSet:
        return ValueSet([self])

    @publish_method('send')
    @publish_method('__next__')
    def _next(self, arguments: Arguments) -> ValueSet:
        return ValueSet.from_sets((lazy_value.infer() for lazy_value in self.py__iter__()))

    def py__stop_iteration_returns(self) -> ValueSet:
        return ValueSet([compiled.builtin_from_name(self.inference_state, 'None')])

    @property
    def name(self) -> AbstractNameDefinition:
        return compiled.CompiledValueName(self, 'Generator')

    def get_annotated_class_object(self) -> Value:
        gen_values = self.merge_types_of_iterate().py__class__()
        gm = TupleGenericManager((gen_values, NO_VALUES, NO_VALUES))
        return self._get_cls().with_generics(gm)

class Generator(GeneratorBase):
    """Handling of `yield` functions."""

    def __init__(self, inference_state: InferenceState, func_execution_context: Context) -> None:
        super().__init__(inference_state)
        self._func_execution_context = func_execution_context

    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyKnownValue, None, None]:
        iterators = self._func_execution_context.infer_annotations()
        if iterators:
            return iterators.iterate(contextualized_node)
        return self._func_execution_context.get_yield_lazy_values()

    def py__stop_iteration_returns(self) -> ValueSet:
        return self._func_execution_context.get_return_values()

    def __repr__(self) -> str:
        return '<%s of %s>' % (type(self).__name__, self._func_execution_context)

def comprehension_from_atom(inference_state: InferenceState, value: Value, atom: Any) -> Union['DictComprehension', 'SetComprehension', 'GeneratorComprehension', 'ListComprehension']:
    bracket = atom.children[0]
    test_list_comp = atom.children[1]
    if bracket == '{':
        if atom.children[1].children[1] == ':':
            sync_comp_for = test_list_comp.children[3]
            if sync_comp_for.type == 'comp_for':
                sync_comp_for = sync_comp_for.children[1]
            return DictComprehension(inference_state, value, sync_comp_for_node=sync_comp_for, key_node=test_list_comp.children[0], value_node=test_list_comp.children[2])
        else:
            cls = SetComprehension
    elif bracket == '(':
        cls = GeneratorComprehension
    elif bracket == '[':
        cls = ListComprehension
    sync_comp_for = test_list_comp.children[1]
    if sync_comp_for.type == 'comp_for':
        sync_comp_for = sync_comp_for.children[1]
    return cls(inference_state, defining_context=value, sync_comp_for_node=sync_comp_for, entry_node=test_list_comp.children[0])

class ComprehensionMixin:

    @inference_state_method_cache()
    def _get_comp_for_context(self, parent_context: Context, comp_for: Any) -> CompForContext:
        return CompForContext(parent_context, comp_for)

    def _nested(self, comp_fors: Tuple[Any, ...], parent_context: Optional[Context] = None) -> Generator[Union[ValueSet, Tuple[ValueSet, ValueSet]], None, None]:
        comp_for = comp_fors[0]
        is_async = comp_for.parent.type == 'comp_for'
        input_node = comp_for.children[3]
        parent_context = parent_context or self._defining_context
        input_types = parent_context.infer_node(input_node)
        cn = ContextualizedNode(parent_context, input_node)
        iterated = input_types.iterate(cn, is_async=is_async)
        exprlist = comp_for.children[1]
        for i, lazy_value in enumerate(iterated):
            types = lazy_value.infer()
            dct = unpack_tuple_to_dict(parent_context, types, exprlist)
            context = self._get_comp_for_context(parent_context, comp_for)
            with context.predefine_names(comp_for, dct):
                try:
                    yield from self._nested(comp_fors[1:], context)
                except IndexError:
                    iterated = context.infer_node(self._entry_node)
                    if self.array_type == 'dict':
                        yield (iterated, context.infer_node(self._value_node))
                    else:
                        yield iterated

    @inference_state_method_cache(default=[])
    @to_list
    def _iterate(self) -> List[Union[ValueSet, Tuple[ValueSet, ValueSet]]]:
        comp_fors = tuple(get_sync_comp_fors(self._sync_comp_for_node))
        yield from self._nested(comp_fors)

    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyKnownValues, None, None]:
        for set_ in self._iterate():
            yield LazyKnownValues(set_)

    def __repr__(self) -> str:
        return '<%s of %s>' % (type(self).__name__, self._sync_comp_for_node)

class _DictMixin:

    def _get_generics(self) -> Tuple[Value, ...]:
        return tuple((c_set.py__class__() for c_set in self.get_mapping_item_values()))

class Sequence(LazyAttributeOverwrite, IterableMixin):
    api_type = 'instance'

    @property
    def name(self) -> AbstractNameDefinition:
        return compiled.CompiledValueName(self, self.array_type)

    def _get_generics(self) -> Tuple[Value, ...]:
        return (self.merge_types_of_iterate().py__class__(),)

    @inference_state_method_cache(default=())
    def _cached_generics(self) -> Tuple[Value, ...]:
        return self._get_generics()

    def _get_wrapped_value(self) -> Value:
        klass = compiled.builtin_from_name(self.inference_state, self.array_type)
        c, = GenericClass(klass, TupleGenericManager(self._cached_generics())).execute_annotation()
        return c

    def py__bool__(self) -> Optional[bool]:
        return None

    @safe_property
    def parent(self) -> Any:
        return self.inference_state.builtins_module

    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: ContextualizedNode) -> ValueSet:
        if self.array_type == 'dict':
            return self._dict_values()
        return iterate_values(ValueSet([self]))

class _BaseComprehension(ComprehensionMixin):

    def __init__(
        self,
        inference_state: InferenceState,
        defining_context: Context,
        sync_comp_for_node: Any,
        entry_node: Any
    ) -> None:
        assert sync_comp_for_node.type == 'sync_comp_for'
        super().__init__(inference_state)
        self._defining_context = defining_context
        self._sync_comp_for_node = sync_comp_for_node
        self._entry_node = entry_node

class ListComprehension(_BaseComprehension, Sequence):
    array_type = 'list'

    def py__simple_getitem__(self, index: Union[int, slice]) -> ValueSet:
        if isinstance(index, slice):
            return ValueSet([self])
        all_types = list(self.py__iter__())
        with reraise_getitem_errors(IndexError, TypeError):
            lazy_value = all_types[index]
        return lazy_value.infer()

class SetComprehension(_BaseComprehension, Sequence):
    array_type = 'set'

class GeneratorComprehension(_BaseComprehension, GeneratorBase):
    pass

class _DictKeyMixin:

    def get_mapping_item_values(self) -> Tuple[ValueSet, ValueSet]:
        return (self._dict_keys(), self._dict_values())

    def get_key_values(self) -> ValueSet:
        return self._dict_keys()

class DictComprehension(ComprehensionMixin, Sequence, _DictKeyMixin):
    array_type = 'dict'

    def __init__(
        self,
        inference_state: InferenceState,
        defining_context: Context,
        sync_comp_for_node: Any,
        key_node: Any,
        value_node: Any
    ) -> None:
        assert sync_comp_for_node.type == 'sync_comp_for'
        super().__init__(inference_state)
        self._defining_context = defining_context
        self._sync_comp_for_node = sync_comp_for_node
        self._entry_node = key_node
        self._value_node = value_node

    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyKnownValues, None, None]:
        for keys, values in self._iterate():
            yield LazyKnownValues(keys)

    def py__simple_getitem__(self, index: Any) -> ValueSet:
        for keys, values in self._iterate():
            for k in keys:
                if k.get_safe_value(default=object()) == index:
                    return values
        raise SimpleGetItemNotFound()

    def _dict_keys(self) -> ValueSet:
        return ValueSet.from_sets((keys for keys, values in self._iterate()))

    def _dict_values(self) -> ValueSet:
        return ValueSet.from_sets((values for keys, values in self._iterate()))

    @publish_method('values')
    def _imitate_values(self, arguments: Arguments) -> ValueSet:
        lazy_value = LazyKnownValues(self._dict_values())
        return ValueSet([FakeList(self.inference_state, [lazy_value])])

    @publish_method('items')
    def _imitate_items(self, arguments: Arguments) -> ValueSet:
        lazy_values = [LazyKnownValue(FakeTuple(self.inference_state, [LazyKnownValues(key), LazyKnownValues(value)])) for key, value in self._iterate()]
        return ValueSet([FakeList(self.inference_state, lazy_values)])

    def exact_key_items(self) -> List:
        return []

class SequenceLiteralValue(Sequence):
    _TUPLE_LIKE = ('testlist_star_expr', 'testlist', 'exprlist')
    mapping = {'(': 'tuple', '[': 'list', '{': 'set'}

    def __init__(self, inference_state: InferenceState, defining_context: Context, atom: Any) -> None:
        super().__init__(inference_state)
        self.atom = atom
        self._defining_context = defining_context
        if self.atom.type in self._TUPLE_LIKE:
            self.array_type = 'tuple'
        else:
            self.array_type = SequenceLiteralValue.mapping[atom.children[0]]
            'The builtin name of the array (list, set, tuple or dict).'

    def _get_generics(self) -> Tuple[Value, ...]:
        if self.array_type == 'tuple':
            return tuple((x.infer().py__class__() for x in self.py__iter__()))
        return super()._get_generics()

    def py__simple_getitem__(self, index: Union[int, str]) -> ValueSet:
        """Here the index is an int/str. Raises IndexError/KeyError."""
        if isinstance(index, slice):
            return ValueSet([self])
        else:
            with reraise_getitem_errors(TypeError, KeyError, IndexError):
                node = self.get_tree_entries()[index]
            if node == ':' or node.type == 'subscript':
                return NO_VALUES
            return self._defining_context.infer_node(node)

    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[Union[LazyKnownValue, LazyTreeValue], None, None]:
        """
        While values returns the possible values for any array field, this
        function returns the value for a certain index.
        """
        for node in self.get_tree_entries():
            if node == ':' or node.type == 'subscript':
                yield LazyKnownValue(Slice(self._defining_context, None, None, None))
            else:
                yield LazyTreeValue(self._defining_context, node)
        yield from check_array_additions(self._defining_context, self)

    def py__len__(self) -> int:
        return len(self.get_tree_entries())

    def get_tree_entries(self) -> List[Any]:
        c = self.atom.children
        if self.atom.type in self._TUPLE_LIKE:
            return c[::2]
        array_node = c[1]
        if array_node in (']', '}', ')'):
            return []
        if array_node.type == 'testlist_comp':
            return [value for value in array_node.children[::2] if value.type != 'star_expr']
        elif array_node.type == 'dictorsetmaker':
            kv = []
            iterator = iter(array_node.children)
            for key in iterator:
                if key == '**':
                    next(iterator)
                    next(iterator, None)
                else:
                    op = next(iterator, None)
                    if op is None or op == ',':
                        if key.type == 'star_expr':
                            pass
                        else:
                            kv.append(key)
                    else:
                        assert op == ':'
                        kv.append((key, next(iterator)))
                        next(iterator, None)
            return kv
        elif array_node.type == 'star_expr':
            return []
        else:
            return [array_node]

    def __repr__(self) -> str:
        return '<%s of %s>' % (self.__class__.__name__, self.atom)

class DictLiteralValue(_DictMixin, SequenceLiteralValue, _DictKeyMixin):
    array_type = 'dict'

    def __init__(self, inference_state: InferenceState, defining_context: Context, atom: Any) -> None:
        Sequence.__init__(self, inference_state)
        self._defining_context = defining_context
        self.atom = atom

    def py__simple_getitem__(self, index: Union[int, str]) -> ValueSet:
        """Here the index is an int/str. Raises IndexError/KeyError."""
        compiled_value_index = compiled.create_simple_object(self.inference_state, index)
        for key, value in self.get_tree_entries():
            for k in self._defining_context.infer_node(key):
                for key_v in k.execute_operation(compiled_value_index, '=='):
                    if key_v.get_safe_value():
                        return self._defining_context.infer_node(value)
        raise SimpleGetItemNotFound('No key found in dictionary %s.' % self)

    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyKnownValues, None, None]:
        """
        While values returns the possible values for any array field, this
        function returns the value for a certain index.
        """
        types = NO_VALUES
        for k, _ in self.get_tree_entries():
            types |= self._defining_context.infer_node(k)
        for _ in types:
            yield LazyKnownValues(types)

    @publish_method('values')
    def _imitate_values(self, arguments: Arguments) -> ValueSet:
        lazy_value = LazyKnownValues(self._dict_values())
        return ValueSet([FakeList(self.inference_state, [lazy_value])])

    @publish_method('items')
    def _imitate_items(self, arguments: Arguments) -> ValueSet:
        lazy_values = [LazyKnownValue(FakeTuple(self.inference_state, (LazyTreeValue(self._defining_context, key_node), LazyTreeValue(self._defining_context, value_node)))) for key_node, value_node in self.get_tree_entries()]
        return ValueSet([FakeList(self.inference_state, lazy_values)]