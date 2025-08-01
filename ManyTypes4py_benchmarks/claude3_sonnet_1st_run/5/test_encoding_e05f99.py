from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union
import pytest
from monkeytype.encoding import CallTraceRow, maybe_decode_type, maybe_encode_type, type_from_dict, type_from_json, type_to_dict, type_to_json, serialize_traces
from mypy_extensions import TypedDict
from monkeytype.exceptions import InvalidTypeError
from monkeytype.tracing import CallTrace
from monkeytype.typing import DUMMY_TYPED_DICT_NAME, NoneType, NotImplementedType, mappingproxy
from .util import Outer
from unittest.mock import Mock

def dummy_func(a: Any, b: Any) -> Any:
    return a + b

class TestTypeConversion:

    @pytest.mark.parametrize('typ', [NoneType, NotImplementedType, mappingproxy, int, Outer, Outer.Inner, Any, Dict, Dict[Any, Any], Dict[int, str], List, List[str], Optional[str], Set[int], Tuple[int, str, str], Tuple, Tuple[()], Type[Outer], Union[Outer.Inner, str, None], Dict[str, Union[str, int]], List[Optional[str]], Dict[str, Union[Dict[str, int], Set[Outer.Inner], Optional[Dict[str, int]]]]])
    def test_type_round_trip(self, typ: Any) -> None:
        assert type_from_dict(type_to_dict(typ)) == typ
        assert type_from_json(type_to_json(typ)) == typ

    @pytest.mark.parametrize('typ, expected', [(Dict[str, int], {'elem_types': [{'module': 'builtins', 'qualname': 'str'}, {'module': 'builtins', 'qualname': 'int'}], 'module': 'typing', 'qualname': 'Dict'}), (TypedDict(DUMMY_TYPED_DICT_NAME, {'a': int, 'b': str}), {'elem_types': {'a': {'module': 'builtins', 'qualname': 'int'}, 'b': {'module': 'builtins', 'qualname': 'str'}}, 'is_typed_dict': True, 'module': 'tests.test_encoding', 'qualname': DUMMY_TYPED_DICT_NAME}), (TypedDict(DUMMY_TYPED_DICT_NAME, {'a': TypedDict(DUMMY_TYPED_DICT_NAME, {'a': int, 'b': str})}), {'elem_types': {'a': {'elem_types': {'a': {'module': 'builtins', 'qualname': 'int'}, 'b': {'module': 'builtins', 'qualname': 'str'}}, 'is_typed_dict': True, 'module': 'tests.test_encoding', 'qualname': DUMMY_TYPED_DICT_NAME}}, 'is_typed_dict': True, 'module': 'tests.test_encoding', 'qualname': DUMMY_TYPED_DICT_NAME})])
    def test_type_to_dict(self, typ: Any, expected: Dict[str, Any]) -> None:
        assert type_to_dict(typ) == expected

    @pytest.mark.parametrize('type_dict, expected', [({'elem_types': {'a': {'module': 'builtins', 'qualname': 'int'}, 'b': {'module': 'builtins', 'qualname': 'str'}}, 'is_typed_dict': True, 'module': 'tests.test_encoding', 'qualname': DUMMY_TYPED_DICT_NAME}, TypedDict(DUMMY_TYPED_DICT_NAME, {'a': int, 'b': str}))])
    def test_type_from_dict(self, type_dict: Dict[str, Any], expected: Any) -> None:
        assert type_from_dict(type_dict) == expected

    @pytest.mark.parametrize('type_dict, expected', [({'elem_types': {'a': {'elem_types': {'a': {'module': 'builtins', 'qualname': 'int'}, 'b': {'module': 'builtins', 'qualname': 'str'}}, 'is_typed_dict': True, 'module': 'tests.test_encoding', 'qualname': DUMMY_TYPED_DICT_NAME}}, 'is_typed_dict': True, 'module': 'tests.test_encoding', 'qualname': DUMMY_TYPED_DICT_NAME}, TypedDict(DUMMY_TYPED_DICT_NAME, {'a': TypedDict(DUMMY_TYPED_DICT_NAME, {'a': int, 'b': str})}))])
    def test_type_from_dict_nested(self, type_dict: Dict[str, Any], expected: Any) -> None:
        assert type_from_dict(type_dict) == expected

    @pytest.mark.parametrize('type_dict, expected', [(TypedDict(DUMMY_TYPED_DICT_NAME, {'a': int, 'b': str}), '{"elem_types": {"a": {"module": "builtins", "qualname": "int"},' + ' "b": {"module": "builtins", "qualname": "str"}},' + ' "is_typed_dict": true, "module": "tests.test_encoding", "qualname": "DUMMY_NAME"}')])
    def test_type_to_json(self, type_dict: Any, expected: str) -> None:
        assert type_to_json(type_dict) == expected

    @pytest.mark.parametrize('type_dict_string, expected', [('{"elem_types": {"a": {"module": "builtins", "qualname": "int"},' + ' "b": {"module": "builtins", "qualname": "str"}},' + ' "is_typed_dict": true, "module": "tests.test_encoding", "qualname": "DUMMY_NAME"}', TypedDict(DUMMY_TYPED_DICT_NAME, {'a': int, 'b': str}))])
    def test_type_from_json(self, type_dict_string: str, expected: Any) -> None:
        assert type_from_json(type_dict_string) == expected

    @pytest.mark.parametrize('type_dict', [TypedDict(DUMMY_TYPED_DICT_NAME, {'a': int, 'b': str})])
    def test_type_round_trip_typed_dict(self, type_dict: Any) -> None:
        assert type_from_dict(type_to_dict(type_dict)) == type_dict
        assert type_from_json(type_to_json(type_dict)) == type_dict

    def test_trace_round_trip(self) -> None:
        trace = CallTrace(dummy_func, {'a': int, 'b': int}, int)
        assert CallTraceRow.from_trace(trace).to_trace() == trace

    def test_convert_non_type(self) -> None:
        with pytest.raises(InvalidTypeError):
            type_from_dict({'module': Outer.Inner.f.__module__, 'qualname': Outer.Inner.f.__qualname__})

    @pytest.mark.parametrize('encoder, typ, expected, should_call_encoder', [(Mock(), None, None, False), (Mock(return_value='foo'), str, 'foo', True)])
    def test_maybe_encode_type(self, encoder: Mock, typ: Optional[Any], expected: Optional[str], should_call_encoder: bool) -> None:
        ret = maybe_encode_type(encoder, typ)
        if should_call_encoder:
            encoder.assert_called_with(typ)
        else:
            encoder.assert_not_called()
        assert ret == expected

    @pytest.mark.parametrize('encoder, typ, expected, should_call_encoder', [(Mock(), None, None, False), (Mock(), 'null', None, False), (Mock(return_value='foo'), 'str', 'foo', True)])
    def test_maybe_decode_type(self, encoder: Mock, typ: Optional[str], expected: Optional[Any], should_call_encoder: bool) -> None:
        ret = maybe_decode_type(encoder, typ)
        if should_call_encoder:
            encoder.assert_called_with(typ)
        else:
            encoder.assert_not_called()
        assert ret == expected

class TestSerializeTraces:

    def test_log_failure_and_continue(self, caplog: Any) -> None:
        traces = [CallTrace(dummy_func, {'a': int, 'b': int}, int), CallTrace(object(), {}), CallTrace(dummy_func, {'a': str, 'b': str}, str)]
        rows = list(serialize_traces(traces))
        expected = [CallTraceRow.from_trace(traces[0]), CallTraceRow.from_trace(traces[2])]
        assert rows == expected
        assert [r.msg for r in caplog.records] == ['Failed to serialize trace']
