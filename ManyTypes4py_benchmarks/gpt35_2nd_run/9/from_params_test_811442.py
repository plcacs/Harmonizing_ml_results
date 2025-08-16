from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union
import pytest
import torch
from allennlp.common import Lazy, Params, Registrable
from allennlp.common.from_params import FromParams, takes_arg, remove_optional, create_kwargs
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import DataLoader, DatasetReader, Tokenizer
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.common.checks import ConfigurationError

class TestFromParams(AllenNlpTestCase):

    def test_takes_arg(self) -> None:

        def bare_function(some_input: int) -> int:
            return some_input + 1
        assert takes_arg(bare_function, 'some_input')
        assert not takes_arg(bare_function, 'some_other_input')

        class SomeClass:
            total: int = 0

            def __init__(self, constructor_param: int) -> None:
                self.constructor_param = constructor_param

            def check_param(self, check: int) -> bool:
                return self.constructor_param == check

            @classmethod
            def set_total(cls, new_total: int) -> None:
                cls.total = new_total
        assert takes_arg(SomeClass, 'self')
        assert takes_arg(SomeClass, 'constructor_param')
        assert not takes_arg(SomeClass, 'check')
        assert takes_arg(SomeClass.check_param, 'check')
        assert not takes_arg(SomeClass.check_param, 'other_check')
        assert takes_arg(SomeClass.set_total, 'new_total')
        assert not takes_arg(SomeClass.set_total, 'total')

    def test_remove_optional(self) -> None:
        optional_type: Optional[Dict[str, str]] = Optional[Dict[str, str]]
        bare_type: Dict[str, str] = remove_optional(optional_type)
        bare_bare_type: Dict[str, str] = remove_optional(bare_type)
        assert bare_type == Dict[str, str]
        assert bare_bare_type == Dict[str, str]
        assert remove_optional(Optional[str]) == str
        assert remove_optional(str) == str

    def test_from_params(self) -> None:
        my_class = MyClass.from_params(Params({'my_int': 10}), my_bool=True)
        assert isinstance(my_class, MyClass)
        assert my_class.my_int == 10
        assert my_class.my_bool

    def test_good_error_message_when_passing_non_params(self) -> None:
        from allennlp.nn import InitializerApplicator
        params = Params({'initializer': [['regex1', 'uniform'], ['regex2', 'orthogonal']]})
        with pytest.raises(ConfigurationError, match='dictionary.*InitializerApplicator'):
            InitializerApplicator.from_params(params=params.pop('initializer'))

    def test_create_kwargs(self) -> None:
        kwargs = create_kwargs(MyClass, MyClass, Params({'my_int': 5}), my_bool=True, my_float=4.4)
        assert kwargs == {'my_int': 5, 'my_bool': True}

    def test_extras(self) -> None:
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int, name: str) -> None:
                self.size = size
                self.name = name

        @A.register('c')
        class C(A):

            def __init__(self, size: int, name: str) -> None:
                self.size = size
                self.name = name

            @classmethod
            def from_params(cls, params: Params, size: int, **extras) -> 'C':
                name = params.pop('name')
                return cls(size=size, name=name)
        params = Params({'type': 'b', 'size': 10})
        b = A.from_params(params, name='extra')
        assert b.name == 'extra'
        assert b.size == 10
        params = Params({'type': 'b', 'size': 10})
        b = A.from_params(params, name='extra', unwanted=True)
        assert b.name == 'extra'
        assert b.size == 10
        params = Params({'type': 'c', 'name': 'extra_c'})
        c = A.from_params(params, size=20)
        assert c.name == 'extra_c'
        assert c.size == 20
        params = Params({'type': 'c', 'name': 'extra_c'})
        c = A.from_params(params, size=20, unwanted=True)
        assert c.name == 'extra_c'
        assert c.size == 20

    def test_extras_for_custom_classes(self) -> None:
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):
            pass

        class BaseClass2(Registrable):
            pass

        @BaseClass.register('A')
        class A(BaseClass):

            def __init__(self, a: int, b: int, val: str) -> None:
                self.a = a
                self.b = b
                self.val = val

            def __hash__(self) -> int:
                return self.b

            def __eq__(self, other: 'A') -> bool:
                return self.b == other.b

            @classmethod
            def from_params(cls, params: Params, a: int, **extras2) -> 'A':
                b = params.pop_int('b')
                val = params.pop('val', 'C')
                params.assert_empty(cls.__name__)
                return cls(a=a, b=b, val=val)

        @BaseClass2.register('B')
        class B(BaseClass2):

            def __init__(self, c: int, b: int) -> None:
                self.c = c
                self.b = b

            @classmethod
            def from_params(cls, params: Params, c: int, **extras2) -> 'B':
                b = params.pop_int('b')
                params.assert_empty(cls.__name__)
                return cls(c=c, b=b)

        @BaseClass.register('E')
        class E(BaseClass):

            def __init__(self, m: int, n: int) -> None:
                self.m = m
                self.n = n

            @classmethod
            def from_params(cls, params: Params, **extras2) -> 'E':
                m = params.pop_int('m')
                params.assert_empty(cls.__name__)
                n = extras2['n']
                return cls(m=m, n=n)

        class C:
            pass

        @BaseClass.register('D')
        class D(BaseClass):

            def __init__(self, arg1, arg2, arg3, arg4, arg5) -> None:
                self.arg1 = arg1
                self.arg2 = arg2
                self.arg3 = arg3
                self.arg4 = arg4
                self.arg5 = arg5
        vals = [1, 2, 3]
        params = Params({'type': 'D', 'arg1': [{'type': 'A', 'b': vals[0]}, {'type': 'A', 'b': vals[1]}, {'type': 'A', 'b': vals[2]}], 'arg2': [{'type': 'A', 'b': vals[0]}, {'type': 'B', 'b': vals[0]}], 'arg3': {'class_1': {'type': 'A', 'b': vals[0]}, 'class_2': {'type': 'A', 'b': vals[1]}}, 'arg4': [{'type': 'A', 'b': vals[0], 'val': 'M'}, {'type': 'A', 'b': vals[1], 'val': 'N'}, {'type': 'A', 'b': vals[1], 'val': 'N'}], 'arg5': [{'type': 'E', 'm': 9}]})
        extra = C()
        tval1 = 5
        tval2 = 6
        d = BaseClass.from_params(params=params, extra=extra, a=tval1, c=tval2, n=10)
        assert len(d.arg1) == len(vals)
        assert isinstance(d.arg1, list)
        assert isinstance(d.arg1[0], A)
        assert all((x.b == y for x, y in zip(d.arg1, vals)))
        assert all((x.a == tval1 for x in d.arg1))
        assert isinstance(d.arg2, tuple)
        assert isinstance(d.arg2[0], A)
        assert isinstance(d.arg2[1], B)
        assert d.arg2[0].a == tval1
        assert d.arg2[1].c == tval2
        assert d.arg2[0].b == d.arg2[1].b == vals[0]
        assert isinstance(d.arg3, dict)
        assert isinstance(d.arg3['class_1'], A)
        assert d.arg3['class_1'].a == d.arg3['class_2'].a == tval1
        assert d.arg3['class_1'].b == vals[0]
        assert d.arg3['class_2'].b == vals[1]
        assert isinstance(d.arg4, set)
        assert len(d.arg4) == 2
        assert any((x.val == 'M' for x in d.arg4))
        assert any((x.val == 'N' for x in d.arg4))
        assert isinstance(d.arg5, list)
        assert isinstance(d.arg5[0], E)
        assert d.arg5[0].m == 9
        assert d.arg5[0].n == 10

    def test_no_constructor(self) -> None:
        params = Params({'type': 'just_spaces'})
        Tokenizer.from_params(params)

    def test_union(self) -> None:

        class A(FromParams):

            def __init__(self, a: Union[int, List[int]]) -> None:
                self.a = a

        class B(FromParams):

            def __init__(self, b: Union[A, List[A]]) -> None:
                self.b = b
        params = Params({'a': 3})
        a = A.from_params(params)
        assert a.a == 3
        params = Params({'a': [3, 4, 5]})
        a = A.from_params(params)
        assert a.a == [3, 4, 5]
        params = Params({'b': {'a': 3}})
        b = B.from_params(params)
        assert isinstance(b.b, A)
        assert b.b.a == 3
        params = Params({'b': [{'a': 3}, {'a': [4, 5]}]})
        b = B.from_params(params)
        assert isinstance(b.b, list)
        assert b.b[0].a == 3
        assert b.b[1].a == [4, 5]

    def test_crazy_nested_union(self) -> None:

        class A(FromParams):

            def __init__(self, a: A) -> None:
                self.a = a

        class B(FromParams):

            def __init__(self, b: B) -> None:
                self.b = b

        class C(FromParams):

            def __init__(self, c: C) -> None:
                self.c = c
        params = Params({'c': {'a': {'a': 3}, 'b': {'a': [4, 5]}})
        c = C.from_params(params)
        assert isinstance(c.c, dict)
        assert c.c['a'].a == 3
        assert c.c['b'].a == [4, 5]

    def test_union_of_castable_types(self) -> None:

        class IntFloat(FromParams):

            def __init__(self, a: Union[int, float]) -> None:
                self.a = a

        class FloatInt(FromParams):

            def __init__(self, a: Union[float, int]) -> None:
                self.a = a
        float_param_str = '{"a": 1.0}'
        int_param_str = '{"a": 1}'
        import json
        for expected_type, param_str in [(int, int_param_str), (float, float_param_str)]:
            for cls in [IntFloat, FloatInt]:
                c = cls.from_params(Params(json.loads(param_str)))
                assert type(c.a) == expected_type

    def test_invalid_type_conversions(self) -> None:

        class A(FromParams):

            def __init__(self, a: int) -> None:
                self.a = a
        with pytest.raises(TypeError):
            A.from_params(Params({'a': '1'}))
        with pytest.raises(TypeError):
            A.from_params(Params({'a': 1.0}))

    def test_dict(self) -> None:
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items: Dict[str, B]) -> None:
                self.items = items
        params = Params({'type': 'd', 'items': {'first': {'type': 'b', 'size': 1}, 'second': {'type': 'b', 'size': 2}}})
        d = C.from_params(params)
        assert isinstance(d.items, dict)
        assert len(d.items) == 2
        assert all((isinstance(key, str) for key in d.items.keys()))
        assert all((isinstance(value, B) for value in d.items.values()))
        assert d.items['first'].size == 1
        assert d.items['second'].size == 2

    def test_dict_not_params(self) -> None:

        class A(FromParams):

            def __init__(self, counts: Dict[str, int]) -> None:
                self.counts = counts
        params = Params({'counts': {'a': 10, 'b': 20}})
        a = A.from_params(params)
        assert isinstance(a.counts, dict)
        assert not isinstance(a.counts, Params)

    def test_list(self) -> None:
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items: List[B]) -> None:
                self.items = items
        params = Params({'type': 'd', 'items': [{'type': 'b', 'size': 1}, {'type': 'b', 'size': 2}]})
        d = C.from_params(params)
        assert isinstance(d.items, list)
        assert len(d.items) == 2
        assert all((isinstance(item, B) for item in d.items))
        assert d.items[0].size == 1
        assert d.items[1].size == 2

    def test_tuple(self) -> None:
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, name: str) -> None:
                self.name = name

        class E(Registrable):
            pass

        @E.register('f')
        class F(E):

            def __init__(self, items: Tuple[B, D]) -> None:
                self.items = items
        params = Params({'type': 'f', 'items': [{'type': 'b', 'size': 1}, {'type': 'd', 'name': 'item2'}]})
        f = E.from_params(params)
        assert isinstance(f.items, tuple)
        assert len(f.items) == 2
        assert isinstance(f.items[0], B)
        assert isinstance(f.items[1], D)
        assert f.items[0].size == 1
        assert f.items[1].name == 'item2'

    def test_set(self) -> None:
        from allennlp.common.registrable import Registrable

        class A(Registrable):

            def __init__(self, name: str) -> None:
                self.name = name

            def __eq__(self, other: 'A') -> bool:
                return self.name == other.name

            def __hash__(self) -> int:
                return hash(self.name)

        @A.register('b')
        class B(A):
            pass

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items: Set[B]) -> None:
                self.items = items
        params = Params({'type': 'd', 'items': [{'type': 'b', 'name': 'item1'}, {'type': 'b', 'name': 'item2'}, {'type': 'b', 'name': 'item2'}]})
        d = C.from_params(params)
        assert isinstance(d.items, set)
        assert len(d.items) == 2
        assert all((isinstance(item, B) for item in d.items))
        assert any((item.name == 'item1' for item in d.items))
        assert any((item.name == 'item2' for item in d.items))

    def test_transferring_of_modules(self) -> None:
        model_archive = str(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        trained_model = load_archive(model_archive).model
        config_file = str(self.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2seq.jsonnet')
        model_params = Params.from_file(config_file).pop('model').as_dict(quiet=True)
        model_params['text_field_embedder'] = {'_pretrained': {'archive_file': model_archive, 'module_path': '_text_field_embedder', 'freeze': True}}
        model_params['seq2seq_encoder'] = {'_pretrained': {'archive_file': model_archive, 'module_path': '_seq2seq_encoder', 'freeze': False}}
        transfer_model = Model.from_params(vocab=trained_model.vocab, params=Params(model_params))
        for trained_parameter, transfer_parameter in zip(trained_model._text_field_embedder.parameters(), transfer_model._text_field_embedder.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        for trained_parameter, transfer_parameter in zip(trained_model._seq2seq_encoder.parameters(), transfer_model._seq2seq_encoder.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        for trained_parameter, transfer_parameter in zip