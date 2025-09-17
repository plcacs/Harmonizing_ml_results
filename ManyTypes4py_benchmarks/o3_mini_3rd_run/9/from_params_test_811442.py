#!/usr/bin/env python3
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union
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

            def __init__(self, constructor_param: Any) -> None:
                self.constructor_param = constructor_param

            def check_param(self, check: Any) -> bool:
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
        optional_type: Any = Optional[Dict[str, str]]
        bare_type: Any = remove_optional(optional_type)
        bare_bare_type: Any = remove_optional(bare_type)
        assert bare_type == Dict[str, str]
        assert bare_bare_type == Dict[str, str]
        assert remove_optional(Optional[str]) == str
        assert remove_optional(str) == str

    def test_from_params(self) -> None:
        my_class: "MyClass" = MyClass.from_params(Params({'my_int': 10}), my_bool=True)
        assert isinstance(my_class, MyClass)
        assert my_class.my_int == 10
        assert my_class.my_bool

    def test_good_error_message_when_passing_non_params(self) -> None:
        from allennlp.nn import InitializerApplicator
        params: Params = Params({'initializer': [['regex1', 'uniform'], ['regex2', 'orthogonal']]})
        with pytest.raises(ConfigurationError, match='dictionary.*InitializerApplicator'):
            InitializerApplicator.from_params(params=params.pop('initializer'))

    def test_create_kwargs(self) -> None:
        kwargs: Dict[str, Any] = create_kwargs(
            MyClass, MyClass, Params({'my_int': 5}), my_bool=True, my_float=4.4
        )
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
            def from_params(cls, params: Params, size: int, **extras: Any) -> "C":
                name: str = params.pop('name')
                return cls(size=size, name=name)

        params: Params = Params({'type': 'b', 'size': 10})
        b: A = A.from_params(params, name='extra')
        assert getattr(b, "name") == 'extra'
        assert getattr(b, "size") == 10
        params = Params({'type': 'b', 'size': 10})
        b = A.from_params(params, name='extra', unwanted=True)
        assert getattr(b, "name") == 'extra'
        assert getattr(b, "size") == 10
        params = Params({'type': 'c', 'name': 'extra_c'})
        c: A = A.from_params(params, size=20)
        assert getattr(c, "name") == 'extra_c'
        assert getattr(c, "size") == 20
        params = Params({'type': 'c', 'name': 'extra_c'})
        c = A.from_params(params, size=20, unwanted=True)
        assert getattr(c, "name") == 'extra_c'
        assert getattr(c, "size") == 20

    def test_extras_for_custom_classes(self) -> None:
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):
            pass

        class BaseClass2(Registrable):
            pass

        @BaseClass.register('A')
        class A(BaseClass):
            def __init__(self, a: Any, b: int, val: str) -> None:
                self.a = a
                self.b = b
                self.val = val

            def __hash__(self) -> int:
                return self.b

            def __eq__(self, other: Any) -> bool:
                return self.b == other.b

            @classmethod
            def from_params(cls, params: Params, a: Any, **extras: Any) -> "A":
                b: int = params.pop_int('b')
                val: str = params.pop('val', 'C')
                params.assert_empty(cls.__name__)
                return cls(a=a, b=b, val=val)

        @BaseClass2.register('B')
        class B(BaseClass2):
            def __init__(self, c: Any, b: int) -> None:
                self.c = c
                self.b = b

            @classmethod
            def from_params(cls, params: Params, c: Any, **extras: Any) -> "B":
                b: int = params.pop_int('b')
                params.assert_empty(cls.__name__)
                return cls(c=c, b=b)

        @BaseClass.register('E')
        class E(BaseClass):
            def __init__(self, m: int, n: int) -> None:
                self.m = m
                self.n = n

            @classmethod
            def from_params(cls, params: Params, **extras2: Any) -> "E":
                m: int = params.pop_int('m')
                params.assert_empty(cls.__name__)
                n: int = extras2['n']
                return cls(m=m, n=n)

        class C:
            pass

        @BaseClass.register('D')
        class D(BaseClass):
            def __init__(self, arg1: Any, arg2: Any, arg3: Any, arg4: Any, arg5: Any) -> None:
                self.arg1 = arg1
                self.arg2 = arg2
                self.arg3 = arg3
                self.arg4 = arg4
                self.arg5 = arg5

        vals: List[int] = [1, 2, 3]
        params = Params({
            'type': 'D',
            'arg1': [{'type': 'A', 'b': vals[0]}, {'type': 'A', 'b': vals[1]}, {'type': 'A', 'b': vals[2]}],
            'arg2': [{'type': 'A', 'b': vals[0]}, {'type': 'B', 'b': vals[0]}],
            'arg3': {'class_1': {'type': 'A', 'b': vals[0]}, 'class_2': {'type': 'A', 'b': vals[1]}},
            'arg4': [{'type': 'A', 'b': vals[0], 'val': 'M'}, {'type': 'A', 'b': vals[1], 'val': 'N'}, {'type': 'A', 'b': vals[1], 'val': 'N'}],
            'arg5': [{'type': 'E', 'm': 9}]
        })
        extra: C = C()
        tval1: int = 5
        tval2: int = 6
        d: D = BaseClass.from_params(params=params, extra=extra, a=tval1, c=tval2, n=10)  # type: ignore
        assert isinstance(d.arg1, list)
        assert len(d.arg1) == len(vals)
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
        params: Params = Params({'type': 'just_spaces'})
        Tokenizer.from_params(params)

    def test_union(self) -> None:
        class A(FromParams):
            def __init__(self, a: Any) -> None:
                self.a = a

        class B(FromParams):
            def __init__(self, b: Any) -> None:
                self.b = b

        params: Params = Params({'a': 3})
        a: A = A.from_params(params)
        assert a.a == 3
        params = Params({'a': [3, 4, 5]})
        a = A.from_params(params)
        assert a.a == [3, 4, 5]
        params = Params({'b': {'a': 3}})
        b: B = B.from_params(params)
        assert isinstance(b.b, A)
        assert b.b.a == 3
        params = Params({'b': [{'a': 3}, {'a': [4, 5]}]})
        b = B.from_params(params)
        assert isinstance(b.b, list)
        assert b.b[0].a == 3
        assert b.b[1].a == [4, 5]

    def test_crazy_nested_union(self) -> None:
        class A(FromParams):
            def __init__(self, a: Any) -> None:
                self.a = a

        class B(FromParams):
            def __init__(self, b: Any) -> None:
                self.b = b

        class C(FromParams):
            def __init__(self, c: Any) -> None:
                self.c = c

        params: Params = Params({'c': {'a': {'a': 3}, 'b': {'a': [4, 5]}}})
        c: C = C.from_params(params)
        assert isinstance(c.c, dict)
        assert c.c['a'].a == 3
        assert c.c['b'].a == [4, 5]

    def test_union_of_castable_types(self) -> None:
        class IntFloat(FromParams):
            def __init__(self, a: Union[int, float]) -> None:
                self.a = a

        class FloatInt(FromParams):
            def __init__(self, a: Union[int, float]) -> None:
                self.a = a

        float_param_str: str = '{"a": 1.0}'
        int_param_str: str = '{"a": 1}'
        import json
        for expected_type, param_str in [(int, int_param_str), (float, float_param_str)]:
            for cls in [IntFloat, FloatInt]:
                c_obj = cls.from_params(Params(json.loads(param_str)))
                assert type(c_obj.a) == expected_type

    def test_invalid_type_conversions(self) -> None:
        class A(FromParams):
            def __init__(self, a: Any) -> None:
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
            def __init__(self, items: Dict[str, Any]) -> None:
                self.items = items

        params: Params = Params({
            'type': 'd',
            'items': {
                'first': {'type': 'b', 'size': 1},
                'second': {'type': 'b', 'size': 2}
            }
        })
        d: D = C.from_params(params)
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

        params: Params = Params({'counts': {'a': 10, 'b': 20}})
        a: A = A.from_params(params)
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
            def __init__(self, items: List[Any]) -> None:
                self.items = items

        params: Params = Params({
            'type': 'd',
            'items': [{'type': 'b', 'size': 1}, {'type': 'b', 'size': 2}]
        })
        d: D = C.from_params(params)
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
            def __init__(self, items: Tuple[Any, ...]) -> None:
                self.items = items

        params: Params = Params({
            'type': 'f',
            'items': [{'type': 'b', 'size': 1}, {'type': 'd', 'name': 'item2'}]
        })
        f: F = E.from_params(params)
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

            def __eq__(self, other: Any) -> bool:
                return self.name == getattr(other, "name", None)

            def __hash__(self) -> int:
                return hash(self.name)

        @A.register('b')
        class B(A):
            pass

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):
            def __init__(self, items: Set[Any]) -> None:
                self.items = items

        params: Params = Params({
            'type': 'd',
            'items': [{'type': 'b', 'name': 'item1'}, {'type': 'b', 'name': 'item2'}, {'type': 'b', 'name': 'item2'}]
        })
        d: D = C.from_params(params)
        assert isinstance(d.items, set)
        assert len(d.items) == 2
        assert all((isinstance(item, B) for item in d.items))
        assert any((item.name == 'item1' for item in d.items))
        assert any((item.name == 'item2' for item in d.items))

    def test_transferring_of_modules(self) -> None:
        model_archive: str = str(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        trained_model: Model = load_archive(model_archive).model
        config_file: str = str(self.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2seq.jsonnet')
        model_params: Dict[str, Any] = Params.from_file(config_file).pop('model').as_dict(quiet=True)
        model_params['text_field_embedder'] = {
            '_pretrained': {
                'archive_file': model_archive,
                'module_path': '_text_field_embedder',
                'freeze': True
            }
        }
        model_params['seq2seq_encoder'] = {
            '_pretrained': {
                'archive_file': model_archive,
                'module_path': '_seq2seq_encoder',
                'freeze': False
            }
        }
        transfer_model: Model = Model.from_params(vocab=trained_model.vocab, params=Params(model_params))
        for trained_parameter, transfer_parameter in zip(trained_model._text_field_embedder.parameters(),
                                                           transfer_model._text_field_embedder.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        for trained_parameter, transfer_parameter in zip(trained_model._seq2seq_encoder.parameters(),
                                                           transfer_model._seq2seq_encoder.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        for trained_parameter, transfer_parameter in zip(trained_model._feedforward.parameters(),
                                                           transfer_model._feedforward.parameters()):
            assert torch.all(trained_parameter != transfer_parameter)
        for parameter in transfer_model._text_field_embedder.parameters():
            assert not parameter.requires_grad
        for parameter in transfer_model._seq2seq_encoder.parameters():
            assert parameter.requires_grad

    def test_transferring_of_modules_ensures_type_consistency(self) -> None:
        model_archive: str = str(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        trained_model: Model = load_archive(model_archive).model
        config_file: str = str(self.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2seq.jsonnet')
        model_params: Dict[str, Any] = Params.from_file(config_file).pop('model').as_dict(quiet=True)
        model_params['text_field_embedder'] = {
            '_pretrained': {
                'archive_file': model_archive,
                'module_path': '_seq2seq_encoder._module'
            }
        }
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=trained_model.vocab, params=Params(model_params))

    def test_bare_string_params(self) -> None:
        reader: DatasetReader = DatasetReader.from_params(Params({'type': 'text_classification_json'}))

        class TestLoader(Registrable):
            @classmethod
            def from_partial_objects(cls, data_loader: DataLoader) -> DataLoader:
                return data_loader.construct(
                    reader=reader,
                    data_path=str(self.FIXTURES_ROOT / 'data' / 'text_classification_json' / 'imdb_corpus2.jsonl')
                )

        TestLoader.register('test', constructor='from_partial_objects')(TestLoader)
        data_loader: DataLoader = TestLoader.from_params(Params({'type': 'test', 'data_loader': {'batch_size': 2}}))
        assert data_loader.batch_size == 2

    def test_kwargs_are_passed_to_superclass(self) -> None:
        params: Params = Params({'type': 'text_classification_json', 'max_instances': 50})
        reader: DatasetReader = DatasetReader.from_params(params)
        assert reader.max_instances == 50

    def test_kwargs_with_multiple_inheritance(self) -> None:
        from numbers import Number

        class A(Registrable):
            def __init__(self, a: Any) -> None:
                self.a = a

        @A.register('b1')
        class B1(A, Number):
            def __init__(self, b: Any, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.b = b

        @A.register('b2')
        class B2(Number, A):
            def __init__(self, b: Any, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.b = b

        b1: Any = B1.from_params(params=Params({'a': 4, 'b': 5}))
        assert b1.b == 5
        assert b1.a == 4
        b2: Any = B2.from_params(params=Params({'a': 4, 'b': 5}))
        assert b2.b == 5
        assert b2.a == 4

    def test_only_infer_superclass_params_if_unknown(self) -> None:
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):
            def __init__(self) -> None:
                self.x: Optional[Any] = None
                self.a: Optional[Any] = None
                self.rest: Optional[Any] = None

        @BaseClass.register('a')
        class A(BaseClass):
            def __init__(self, a: Any, x: Any, **kwargs: Any) -> None:
                super().__init__()
                self.x = x
                self.a = a
                self.rest = kwargs

        @BaseClass.register('b')
        class B(A):
            def __init__(self, a: Any, x: int = 42, **kwargs: Any) -> None:
                super().__init__(x=x, a=-1, raw_a=a, **kwargs)

        params: Params = Params({'type': 'b', 'a': '123'})
        instance: BaseClass = BaseClass.from_params(params)
        assert instance.x == 42
        assert instance.a == -1
        assert len(instance.rest) == 1  # type: ignore
        assert type(instance.rest['raw_a']) == str  # type: ignore
        assert instance.rest['raw_a'] == '123'  # type: ignore

    def test_kwargs_are_passed_to_deeper_superclasses(self) -> None:
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):
            def __init__(self) -> None:
                self.a: Optional[Any] = None
                self.b: Optional[Any] = None
                self.c: Optional[Any] = None

        @BaseClass.register('a')
        class A(BaseClass):
            def __init__(self, a: Any) -> None:
                super().__init__()
                self.a = a

        @BaseClass.register('b')
        class B(A):
            def __init__(self, b: Any, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.b = b

        @BaseClass.register('c')
        class C(B):
            def __init__(self, c: Any, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.c = c

        params: Params = Params({'type': 'c', 'a': 'a_value', 'b': 'b_value', 'c': 'c_value'})
        instance: BaseClass = BaseClass.from_params(params)
        assert instance.a == 'a_value'
        assert instance.b == 'b_value'
        assert instance.c == 'c_value'

    def test_lazy_construction_can_happen_multiple_times(self) -> None:
        test_string: str = 'this is a test'
        extra_string: str = 'extra string'

        class ConstructedObject(FromParams):
            def __init__(self, string: str, extra: str) -> None:
                self.string = string
                self.extra = extra

        class Testing(FromParams):
            def __init__(self, lazy_object: Lazy[ConstructedObject]) -> None:
                first_time: ConstructedObject = lazy_object.construct(extra=extra_string)
                second_time: ConstructedObject = lazy_object.construct(extra=extra_string)
                assert first_time.string == test_string
                assert first_time.extra == extra_string
                assert second_time.string == test_string
                assert second_time.extra == extra_string

        Testing.from_params(Params({'lazy_object': {'string': test_string}}))

    def test_lazy_and_from_params_can_be_pickled(self) -> None:
        import pickle
        baz: Baz = Baz.from_params(Params({'bar': {'foo': {'a': 2}}}))
        pickle.dumps(baz)

    def test_optional_vs_required_lazy_objects(self) -> None:
        class ConstructedObject(FromParams):
            def __init__(self, a: int) -> None:
                self.a = a

        class Testing(FromParams):
            def __init__(
                self,
                lazy1: Lazy[ConstructedObject],
                lazy2: Lazy[ConstructedObject] = Lazy(ConstructedObject),
                lazy3: Optional[Lazy[ConstructedObject]] = None,
                lazy4: Lazy[ConstructedObject] = Lazy(ConstructedObject)
            ) -> None:
                self.lazy1 = lazy1.construct()
                self.lazy2 = lazy2.construct(a=2)
                self.lazy3 = None if lazy3 is None else lazy3.construct()
                self.lazy4 = None if lazy4 is None else lazy4.construct(a=1)

        test1: Testing = Testing.from_params(Params({'lazy1': {'a': 1}}))
        assert test1.lazy1.a == 1
        assert test1.lazy2.a == 2
        assert test1.lazy3 is None
        assert test1.lazy4 is not None
        test2: Testing = Testing.from_params(Params({'lazy1': {'a': 1}, 'lazy2': {'a': 3}}))
        assert test2.lazy1.a == 1
        assert test2.lazy2.a == 3
        assert test2.lazy3 is None
        assert test2.lazy4 is not None
        test3: Testing = Testing.from_params(
            Params({'lazy1': {'a': 1}, 'lazy3': {'a': 3}, 'lazy4': None})
        )
        assert test3.lazy1.a == 1
        assert test3.lazy2.a == 2
        assert test3.lazy3 is not None
        assert test3.lazy3.a == 3
        assert test3.lazy4 is None
        with pytest.raises(ConfigurationError, match='key "lazy1" is required'):
            Testing.from_params(Params({}))

    def test_iterable(self) -> None:
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
            def __init__(self, items: Iterable[Any]) -> None:
                self.items = items

        params: Params = Params({
            'type': 'd',
            'items': [{'type': 'b', 'size': 1}, {'type': 'b', 'size': 2}]
        })
        d: D = C.from_params(params)
        assert isinstance(d.items, Iterable)
        items: List[Any] = list(d.items)  # type: ignore
        assert len(items) == 2
        assert all((isinstance(item, B) for item in items))
        assert items[0].size == 1
        assert items[1].size == 2

    def test_mapping(self) -> None:
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
            def __init__(self, items: Mapping[str, Any]) -> None:
                self.items = items

        params: Params = Params({
            'type': 'd',
            'items': {
                'first': {'type': 'b', 'size': 1},
                'second': {'type': 'b', 'size': 2}
            }
        })
        d: D = C.from_params(params)
        assert isinstance(d.items, Mapping)
        assert len(d.items) == 2
        assert all((isinstance(key, str) for key in d.items.keys()))
        assert all((isinstance(value, B) for value in d.items.values()))
        assert d.items['first'].size == 1
        assert d.items['second'].size == 2

    def test_extra_parameters_are_not_allowed_when_there_is_no_constructor(self) -> None:
        class A(FromParams):
            pass

        with pytest.raises(ConfigurationError, match='Extra parameters'):
            A.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))

    def test_explicit_kwargs_always_passed_to_constructor(self) -> None:
        class Base(FromParams):
            def __init__(self, lazy: bool = False, x: int = 0) -> None:
                self.lazy = lazy
                self.x = x

        class A(Base):
            def __init__(self, **kwargs: Any) -> None:
                assert 'lazy' in kwargs
                super().__init__(**kwargs)

        A.from_params(Params({'lazy': False}))

        class B(Base):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(lazy=True, **kwargs)

        b: B = B.from_params(Params({}))
        assert b.lazy is True

    def test_raises_when_there_are_no_implementations(self) -> None:
        class A(Registrable):
            pass

        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params('nonexistent_class')
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params(Params({}))

        class B(Registrable):
            def __init__(self) -> None:
                pass

        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params('nonexistent_class')
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params(Params({}))

    def test_from_params_raises_error_on_wrong_parameter_name_in_optional_union(self) -> None:
        class NestedClass(FromParams):
            def __init__(self, varname: Optional[str] = None) -> None:
                self.varname = varname

        class WrapperClass(FromParams):
            def __init__(self, nested_class: Optional[Any] = None) -> None:
                if isinstance(nested_class, str):
                    nested_class = NestedClass(varname=nested_class)
                self.nested_class = nested_class

        with pytest.raises(ConfigurationError):
            WrapperClass.from_params(params=Params({'nested_class': {'wrong_varname': 'varstring'}}))

    def test_from_params_handles_base_class_kwargs(self) -> None:
        class Foo(FromParams):
            def __init__(self, a: Any, b: Optional[Any] = None, **kwargs: Any) -> None:
                self.a = a
                self.b = b
                for key, value in kwargs.items():
                    setattr(self, key, value)

        foo: Foo = Foo.from_params(Params({'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        foo = Foo.from_params(Params({'a': 2, 'b': 'hi', 'c': {'2': '3'}}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == {'2': '3'}

        class Bar(Foo):
            def __init__(self, a: Any, b: Any, d: Any, **kwargs: Any) -> None:
                super().__init__(a, b=b, **kwargs)
                self.d = d

        bar: Bar = Bar.from_params(Params({'a': 2, 'b': 'hi', 'c': {'2': '3'}, 'd': 0}))
        assert bar.a == 2
        assert bar.b == 'hi'
        assert bar.c == {'2': '3'}
        assert bar.d == 0

        class Baz(Foo):
            def __init__(self, a: Any, b: str = 'a', **kwargs: Any) -> None:
                super().__init__(a, b=b, **kwargs)

        baz: Baz = Baz.from_params(Params({'a': 2, 'b': None}))
        assert baz.b is None
        baz = Baz.from_params(Params({'a': 2}))
        assert baz.b == 'a'

    def test_from_params_base_class_kwargs_crashes_if_params_not_handled(self) -> None:
        class Bar(FromParams):
            def __init__(self, c: Optional[Any] = None) -> None:
                self.c = c

        class Foo(Bar):
            def __init__(self, a: Any, b: Optional[Any] = None, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.a = a
                self.b = b

        foo: Foo = Foo.from_params(Params({'a': 2, 'b': 'hi', 'c': 'some value'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == 'some value'
        with pytest.raises(TypeError, match='invalid_key'):
            Foo.from_params(Params({'a': 2, 'b': 'hi', 'invalid_key': 'some value'}))

    def test_from_params_handles_kwargs_in_non_from_params_registered_class(self) -> None:
        class Bar(Registrable):
            pass

        class Baz:
            def __init__(self, a: Any) -> None:
                self.a = a

        @Bar.register('foo')
        class Foo(Baz):
            def __init__(self, a: Any, b: Optional[Any] = None, **kwargs: Any) -> None:
                super().__init__(a)
                self.b = b
                for key, value in kwargs.items():
                    setattr(self, key, value)

        foo: Baz = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert getattr(foo, "b") == 'hi'
        foo = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi', 'c': {'2': '3'}}))
        assert foo.a == 2
        assert getattr(foo, "b") == 'hi'
        assert getattr(foo, "c") == {'2': '3'}

    def test_from_params_does_not_pass_extras_to_non_from_params_registered_class(self) -> None:
        class Bar(Registrable):
            pass

        class Baz:
            def __init__(self, a: Any, c: Optional[Any] = None) -> None:
                self.a = a
                self.c = c

        @Bar.register('foo')
        class Foo(Baz):
            def __init__(self, a: Any, b: Optional[Any] = None, **kwargs: Any) -> None:
                super().__init__(a, **kwargs)
                self.b = b

        foo: Baz = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert getattr(foo, "b") == 'hi'
        assert foo.c is None
        foo = Bar.from_params(params=Params({'type': 'foo', 'a': 2, 'b': 'hi', 'c': {'2': '3'}}), extra='4')
        assert foo.a == 2
        assert getattr(foo, "b") == 'hi'
        assert foo.c == {'2': '3'}

    def test_from_params_child_has_kwargs_base_implicit_constructor(self) -> None:
        class Foo(FromParams):
            pass

        class Bar(Foo):
            def __init__(self, a: Any, **kwargs: Any) -> None:
                self.a = a

        bar: Bar = Bar.from_params(Params({'a': 2}))
        assert bar.a == 2

    def test_from_params_has_args(self) -> None:
        class Foo(FromParams):
            def __init__(self, a: Any, *args: Any) -> None:
                self.a = a

        foo: Foo = Foo.from_params(Params({'a': 2}))
        assert foo.a == 2


class MyClass(FromParams):
    def __init__(self, my_int: int, my_bool: bool = False) -> None:
        self.my_int = my_int
        self.my_bool = my_bool


class Foo(FromParams):
    def __init__(self, a: int = 1) -> None:
        self.a = a


class Bar(FromParams):
    def __init__(self, foo: Foo) -> None:
        self.foo = foo


class Baz(FromParams):
    def __init__(self, bar: Lazy[Bar]) -> None:
        self._bar = bar

    @property
    def bar(self) -> Bar:
        return self._bar.construct()