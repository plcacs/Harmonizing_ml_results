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

    def test_takes_arg(self):

        def bare_function(some_input):
            return some_input + 1
        assert takes_arg(bare_function, 'some_input')
        assert not takes_arg(bare_function, 'some_other_input')

        class SomeClass:
            total = 0

            def __init__(self, constructor_param):
                self.constructor_param = constructor_param

            def check_param(self, check):
                return self.constructor_param == check

            @classmethod
            def set_total(cls, new_total):
                cls.total = new_total
        assert takes_arg(SomeClass, 'self')
        assert takes_arg(SomeClass, 'constructor_param')
        assert not takes_arg(SomeClass, 'check')
        assert takes_arg(SomeClass.check_param, 'check')
        assert not takes_arg(SomeClass.check_param, 'other_check')
        assert takes_arg(SomeClass.set_total, 'new_total')
        assert not takes_arg(SomeClass.set_total, 'total')

    def test_remove_optional(self):
        optional_type = Optional[Dict[str, str]]
        bare_type = remove_optional(optional_type)
        bare_bare_type = remove_optional(bare_type)
        assert bare_type == Dict[str, str]
        assert bare_bare_type == Dict[str, str]
        assert remove_optional(Optional[str]) == str
        assert remove_optional(str) == str

    def test_from_params(self):
        my_class = MyClass.from_params(Params({'my_int': 10}), my_bool=True)
        assert isinstance(my_class, MyClass)
        assert my_class.my_int == 10
        assert my_class.my_bool

    def test_good_error_message_when_passing_non_params(self):
        from allennlp.nn import InitializerApplicator
        params = Params({'initializer': [['regex1', 'uniform'], ['regex2', 'orthogonal']]})
        with pytest.raises(ConfigurationError, match='dictionary.*InitializerApplicator'):
            InitializerApplicator.from_params(params=params.pop('initializer'))

    def test_create_kwargs(self):
        kwargs = create_kwargs(MyClass, MyClass, Params({'my_int': 5}), my_bool=True, my_float=4.4)
        assert kwargs == {'my_int': 5, 'my_bool': True}

    def test_extras(self):
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size, name):
                self.size = size
                self.name = name

        @A.register('c')
        class C(A):

            def __init__(self, size, name):
                self.size = size
                self.name = name

            @classmethod
            def from_params(cls, params, size, **extras):
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

    def test_extras_for_custom_classes(self):
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):
            pass

        class BaseClass2(Registrable):
            pass

        @BaseClass.register('A')
        class A(BaseClass):

            def __init__(self, a, b, val):
                self.a = a
                self.b = b
                self.val = val

            def __hash__(self):
                return self.b

            def __eq__(self, other):
                return self.b == other.b

            @classmethod
            def from_params(cls, params, a, **extras):
                b = params.pop_int('b')
                val = params.pop('val', 'C')
                params.assert_empty(cls.__name__)
                return cls(a=a, b=b, val=val)

        @BaseClass2.register('B')
        class B(BaseClass2):

            def __init__(self, c, b):
                self.c = c
                self.b = b

            @classmethod
            def from_params(cls, params, c, **extras):
                b = params.pop_int('b')
                params.assert_empty(cls.__name__)
                return cls(c=c, b=b)

        @BaseClass.register('E')
        class E(BaseClass):

            def __init__(self, m, n):
                self.m = m
                self.n = n

            @classmethod
            def from_params(cls, params, **extras2):
                m = params.pop_int('m')
                params.assert_empty(cls.__name__)
                n = extras2['n']
                return cls(m=m, n=n)

        class C:
            pass

        @BaseClass.register('D')
        class D(BaseClass):

            def __init__(self, arg1, arg2, arg3, arg4, arg5):
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

    def test_no_constructor(self):
        params = Params({'type': 'just_spaces'})
        Tokenizer.from_params(params)

    def test_union(self):

        class A(FromParams):

            def __init__(self, a):
                self.a = a

        class B(FromParams):

            def __init__(self, b):
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

    def test_crazy_nested_union(self):

        class A(FromParams):

            def __init__(self, a):
                self.a = a

        class B(FromParams):

            def __init__(self, b):
                self.b = b

        class C(FromParams):

            def __init__(self, c):
                self.c = c
        params = Params({'c': {'a': {'a': 3}, 'b': {'a': [4, 5]}}})
        c = C.from_params(params)
        assert isinstance(c.c, dict)
        assert c.c['a'].a == 3
        assert c.c['b'].a == [4, 5]

    def test_union_of_castable_types(self):

        class IntFloat(FromParams):

            def __init__(self, a):
                self.a = a

        class FloatInt(FromParams):

            def __init__(self, a):
                self.a = a
        float_param_str = '{"a": 1.0}'
        int_param_str = '{"a": 1}'
        import json
        for expected_type, param_str in [(int, int_param_str), (float, float_param_str)]:
            for cls in [IntFloat, FloatInt]:
                c = cls.from_params(Params(json.loads(param_str)))
                assert type(c.a) == expected_type

    def test_invalid_type_conversions(self):

        class A(FromParams):

            def __init__(self, a):
                self.a = a
        with pytest.raises(TypeError):
            A.from_params(Params({'a': '1'}))
        with pytest.raises(TypeError):
            A.from_params(Params({'a': 1.0}))

    def test_dict(self):
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size):
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items):
                self.items = items
        params = Params({'type': 'd', 'items': {'first': {'type': 'b', 'size': 1}, 'second': {'type': 'b', 'size': 2}}})
        d = C.from_params(params)
        assert isinstance(d.items, dict)
        assert len(d.items) == 2
        assert all((isinstance(key, str) for key in d.items.keys()))
        assert all((isinstance(value, B) for value in d.items.values()))
        assert d.items['first'].size == 1
        assert d.items['second'].size == 2

    def test_dict_not_params(self):

        class A(FromParams):

            def __init__(self, counts):
                self.counts = counts
        params = Params({'counts': {'a': 10, 'b': 20}})
        a = A.from_params(params)
        assert isinstance(a.counts, dict)
        assert not isinstance(a.counts, Params)

    def test_list(self):
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size):
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items):
                self.items = items
        params = Params({'type': 'd', 'items': [{'type': 'b', 'size': 1}, {'type': 'b', 'size': 2}]})
        d = C.from_params(params)
        assert isinstance(d.items, list)
        assert len(d.items) == 2
        assert all((isinstance(item, B) for item in d.items))
        assert d.items[0].size == 1
        assert d.items[1].size == 2

    def test_tuple(self):
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size):
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, name):
                self.name = name

        class E(Registrable):
            pass

        @E.register('f')
        class F(E):

            def __init__(self, items):
                self.items = items
        params = Params({'type': 'f', 'items': [{'type': 'b', 'size': 1}, {'type': 'd', 'name': 'item2'}]})
        f = E.from_params(params)
        assert isinstance(f.items, tuple)
        assert len(f.items) == 2
        assert isinstance(f.items[0], B)
        assert isinstance(f.items[1], D)
        assert f.items[0].size == 1
        assert f.items[1].name == 'item2'

    def test_set(self):
        from allennlp.common.registrable import Registrable

        class A(Registrable):

            def __init__(self, name):
                self.name = name

            def __eq__(self, other):
                return self.name == other.name

            def __hash__(self):
                return hash(self.name)

        @A.register('b')
        class B(A):
            pass

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items):
                self.items = items
        params = Params({'type': 'd', 'items': [{'type': 'b', 'name': 'item1'}, {'type': 'b', 'name': 'item2'}, {'type': 'b', 'name': 'item2'}]})
        d = C.from_params(params)
        assert isinstance(d.items, set)
        assert len(d.items) == 2
        assert all((isinstance(item, B) for item in d.items))
        assert any((item.name == 'item1' for item in d.items))
        assert any((item.name == 'item2' for item in d.items))

    def test_transferring_of_modules(self):
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
        for trained_parameter, transfer_parameter in zip(trained_model._feedforward.parameters(), transfer_model._feedforward.parameters()):
            assert torch.all(trained_parameter != transfer_parameter)
        for parameter in transfer_model._text_field_embedder.parameters():
            assert not parameter.requires_grad
        for parameter in transfer_model._seq2seq_encoder.parameters():
            assert parameter.requires_grad

    def test_transferring_of_modules_ensures_type_consistency(self):
        model_archive = str(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        trained_model = load_archive(model_archive).model
        config_file = str(self.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2seq.jsonnet')
        model_params = Params.from_file(config_file).pop('model').as_dict(quiet=True)
        model_params['text_field_embedder'] = {'_pretrained': {'archive_file': model_archive, 'module_path': '_seq2seq_encoder._module'}}
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=trained_model.vocab, params=Params(model_params))

    def test_bare_string_params(self):
        reader = DatasetReader.from_params(Params({'type': 'text_classification_json'}))

        class TestLoader(Registrable):

            @classmethod
            def from_partial_objects(cls, data_loader):
                return data_loader.construct(reader=reader, data_path=str(self.FIXTURES_ROOT / 'data' / 'text_classification_json' / 'imdb_corpus2.jsonl'))
        TestLoader.register('test', constructor='from_partial_objects')(TestLoader)
        data_loader = TestLoader.from_params(Params({'type': 'test', 'data_loader': {'batch_size': 2}}))
        assert data_loader.batch_size == 2

    def test_kwargs_are_passed_to_superclass(self):
        params = Params({'type': 'text_classification_json', 'max_instances': 50})
        reader = DatasetReader.from_params(params)
        assert reader.max_instances == 50

    def test_kwargs_with_multiple_inheritance(self):

        class A(Registrable):

            def __init__(self, a):
                self.a = a
        from numbers import Number

        @A.register('b1')
        class B1(A, Number):

            def __init__(self, b, **kwargs):
                super().__init__(**kwargs)
                self.b = b

        @A.register('b2')
        class B2(Number, A):

            def __init__(self, b, **kwargs):
                super().__init__(**kwargs)
                self.b = b
        b = B1.from_params(params=Params({'a': 4, 'b': 5}))
        assert b.b == 5
        assert b.a == 4
        b = B2.from_params(params=Params({'a': 4, 'b': 5}))
        assert b.b == 5
        assert b.a == 4

    def test_only_infer_superclass_params_if_unknown(self):
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):

            def __init__(self):
                self.x = None
                self.a = None
                self.rest = None

        @BaseClass.register('a')
        class A(BaseClass):

            def __init__(self, a, x, **kwargs):
                super().__init__()
                self.x = x
                self.a = a
                self.rest = kwargs

        @BaseClass.register('b')
        class B(A):

            def __init__(self, a, x=42, **kwargs):
                super().__init__(x=x, a=-1, raw_a=a, **kwargs)
        params = Params({'type': 'b', 'a': '123'})
        instance = BaseClass.from_params(params)
        assert instance.x == 42
        assert instance.a == -1
        assert len(instance.rest) == 1
        assert type(instance.rest['raw_a']) == str
        assert instance.rest['raw_a'] == '123'

    def test_kwargs_are_passed_to_deeper_superclasses(self):
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):

            def __init__(self):
                self.a = None
                self.b = None
                self.c = None

        @BaseClass.register('a')
        class A(BaseClass):

            def __init__(self, a):
                super().__init__()
                self.a = a

        @BaseClass.register('b')
        class B(A):

            def __init__(self, b, **kwargs):
                super().__init__(**kwargs)
                self.b = b

        @BaseClass.register('c')
        class C(B):

            def __init__(self, c, **kwargs):
                super().__init__(**kwargs)
                self.c = c
        params = Params({'type': 'c', 'a': 'a_value', 'b': 'b_value', 'c': 'c_value'})
        instance = BaseClass.from_params(params)
        assert instance.a == 'a_value'
        assert instance.b == 'b_value'
        assert instance.c == 'c_value'

    def test_lazy_construction_can_happen_multiple_times(self):
        test_string = 'this is a test'
        extra_string = 'extra string'

        class ConstructedObject(FromParams):

            def __init__(self, string, extra):
                self.string = string
                self.extra = extra

        class Testing(FromParams):

            def __init__(self, lazy_object):
                first_time = lazy_object.construct(extra=extra_string)
                second_time = lazy_object.construct(extra=extra_string)
                assert first_time.string == test_string
                assert first_time.extra == extra_string
                assert second_time.string == test_string
                assert second_time.extra == extra_string
        Testing.from_params(Params({'lazy_object': {'string': test_string}}))

    def test_lazy_and_from_params_can_be_pickled(self):
        import pickle
        baz = Baz.from_params(Params({'bar': {'foo': {'a': 2}}}))
        pickle.dumps(baz)

    def test_optional_vs_required_lazy_objects(self):

        class ConstructedObject(FromParams):

            def __init__(self, a):
                self.a = a

        class Testing(FromParams):

            def __init__(self, lazy1, lazy2=Lazy(ConstructedObject), lazy3=None, lazy4=Lazy(ConstructedObject)):
                self.lazy1 = lazy1.construct()
                self.lazy2 = lazy2.construct(a=2)
                self.lazy3 = None if lazy3 is None else lazy3.construct()
                self.lazy4 = None if lazy4 is None else lazy4.construct(a=1)
        test1 = Testing.from_params(Params({'lazy1': {'a': 1}}))
        assert test1.lazy1.a == 1
        assert test1.lazy2.a == 2
        assert test1.lazy3 is None
        assert test1.lazy4 is not None
        test2 = Testing.from_params(Params({'lazy1': {'a': 1}, 'lazy2': {'a': 3}}))
        assert test2.lazy1.a == 1
        assert test2.lazy2.a == 3
        assert test2.lazy3 is None
        assert test2.lazy4 is not None
        test3 = Testing.from_params(Params({'lazy1': {'a': 1}, 'lazy3': {'a': 3}, 'lazy4': None}))
        assert test3.lazy1.a == 1
        assert test3.lazy2.a == 2
        assert test3.lazy3 is not None
        assert test3.lazy3.a == 3
        assert test3.lazy4 is None
        with pytest.raises(ConfigurationError, match='key "lazy1" is required'):
            Testing.from_params(Params({}))

    def test_iterable(self):
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size):
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items):
                self.items = items
        params = Params({'type': 'd', 'items': [{'type': 'b', 'size': 1}, {'type': 'b', 'size': 2}]})
        d = C.from_params(params)
        assert isinstance(d.items, Iterable)
        items = list(d.items)
        assert len(items) == 2
        assert all((isinstance(item, B) for item in items))
        assert items[0].size == 1
        assert items[1].size == 2

    def test_mapping(self):
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size):
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items):
                self.items = items
        params = Params({'type': 'd', 'items': {'first': {'type': 'b', 'size': 1}, 'second': {'type': 'b', 'size': 2}}})
        d = C.from_params(params)
        assert isinstance(d.items, Mapping)
        assert len(d.items) == 2
        assert all((isinstance(key, str) for key in d.items.keys()))
        assert all((isinstance(value, B) for value in d.items.values()))
        assert d.items['first'].size == 1
        assert d.items['second'].size == 2

    def test_extra_parameters_are_not_allowed_when_there_is_no_constructor(self):

        class A(FromParams):
            pass
        with pytest.raises(ConfigurationError, match='Extra parameters'):
            A.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))

    def test_explicit_kwargs_always_passed_to_constructor(self):

        class Base(FromParams):

            def __init__(self, lazy=False, x=0):
                self.lazy = lazy
                self.x = x

        class A(Base):

            def __init__(self, **kwargs):
                assert 'lazy' in kwargs
                super().__init__(**kwargs)
        A.from_params(Params({'lazy': False}))

        class B(Base):

            def __init__(self, **kwargs):
                super().__init__(lazy=True, **kwargs)
        b = B.from_params(Params({}))
        assert b.lazy is True

    def test_raises_when_there_are_no_implementations(self):

        class A(Registrable):
            pass
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params('nonexistent_class')
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params(Params({}))

        class B(Registrable):

            def __init__(self):
                pass
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params('nonexistent_class')
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params(Params({}))

    def test_from_params_raises_error_on_wrong_parameter_name_in_optional_union(self):

        class NestedClass(FromParams):

            def __init__(self, varname=None):
                self.varname = varname

        class WrapperClass(FromParams):

            def __init__(self, nested_class=None):
                if isinstance(nested_class, str):
                    nested_class = NestedClass(varname=nested_class)
                self.nested_class = nested_class
        with pytest.raises(ConfigurationError):
            WrapperClass.from_params(params=Params({'nested_class': {'wrong_varname': 'varstring'}}))

    def test_from_params_handles_base_class_kwargs(self):

        class Foo(FromParams):

            def __init__(self, a, b=None, **kwargs):
                self.a = a
                self.b = b
                for key, value in kwargs.items():
                    setattr(self, key, value)
        foo = Foo.from_params(Params({'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        foo = Foo.from_params(Params({'a': 2, 'b': 'hi', 'c': {'2': '3'}}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == {'2': '3'}

        class Bar(Foo):

            def __init__(self, a, b, d, **kwargs):
                super().__init__(a, b=b, **kwargs)
                self.d = d
        bar = Bar.from_params(Params({'a': 2, 'b': 'hi', 'c': {'2': '3'}, 'd': 0}))
        assert bar.a == 2
        assert bar.b == 'hi'
        assert bar.c == {'2': '3'}
        assert bar.d == 0

        class Baz(Foo):

            def __init__(self, a, b='a', **kwargs):
                super().__init__(a, b=b, **kwargs)
        baz = Baz.from_params(Params({'a': 2, 'b': None}))
        assert baz.b is None
        baz = Baz.from_params(Params({'a': 2}))
        assert baz.b == 'a'

    def test_from_params_base_class_kwargs_crashes_if_params_not_handled(self):

        class Bar(FromParams):

            def __init__(self, c=None):
                self.c = c

        class Foo(Bar):

            def __init__(self, a, b=None, **kwargs):
                super().__init__(**kwargs)
                self.a = a
                self.b = b
        foo = Foo.from_params(Params({'a': 2, 'b': 'hi', 'c': 'some value'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == 'some value'
        with pytest.raises(TypeError, match='invalid_key'):
            Foo.from_params(Params({'a': 2, 'b': 'hi', 'invalid_key': 'some value'}))

    def test_from_params_handles_kwargs_in_non_from_params_registered_class(self):

        class Bar(Registrable):
            pass

        class Baz:

            def __init__(self, a):
                self.a = a

        @Bar.register('foo')
        class Foo(Baz):

            def __init__(self, a, b=None, **kwargs):
                super().__init__(a)
                self.b = b
                for key, value in kwargs.items():
                    setattr(self, key, value)
        foo = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        foo = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi', 'c': {'2': '3'}}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == {'2': '3'}

    def test_from_params_does_not_pass_extras_to_non_from_params_registered_class(self):

        class Bar(Registrable):
            pass

        class Baz:

            def __init__(self, a, c=None):
                self.a = a
                self.c = c

        @Bar.register('foo')
        class Foo(Baz):

            def __init__(self, a, b=None, **kwargs):
                super().__init__(a, **kwargs)
                self.b = b
        foo = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c is None
        foo = Bar.from_params(params=Params({'type': 'foo', 'a': 2, 'b': 'hi', 'c': {'2': '3'}}), extra='4')
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == {'2': '3'}

    def test_from_params_child_has_kwargs_base_implicit_constructor(self):

        class Foo(FromParams):
            pass

        class Bar(Foo):

            def __init__(self, a, **kwargs):
                self.a = a
        bar = Bar.from_params(Params({'a': 2}))
        assert bar.a == 2

    def test_from_params_has_args(self):

        class Foo(FromParams):

            def __init__(self, a, *args):
                self.a = a
        foo = Foo.from_params(Params({'a': 2}))
        assert foo.a == 2

class MyClass(FromParams):

    def __init__(self, my_int, my_bool=False):
        self.my_int = my_int
        self.my_bool = my_bool

class Foo(FromParams):

    def __init__(self, a=1):
        self.a = a

class Bar(FromParams):

    def __init__(self, foo):
        self.foo = foo

class Baz(FromParams):

    def __init__(self, bar):
        self._bar = bar

    @property
    def bar(self):
        return self._bar.construct()