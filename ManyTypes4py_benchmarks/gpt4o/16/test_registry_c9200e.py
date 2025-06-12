import typing
from decimal import Decimal
import faust
import pytest
from faust.exceptions import KeyDecodeError, ValueDecodeError
from faust.utils import json
from mode.utils.mocks import Mock

class Case(typing.NamedTuple):
    payload: typing.Any
    typ: typing.Optional[typing.Type]
    serializer: str
    expected: typing.Any

class User(faust.Record):
    id: str
    first_name: str
    last_name: str

class Account(faust.Record):
    id: str
    active: bool
    user: User

USER1 = User('A2', 'George', 'Costanza')
ACCOUNT1 = Account(id='A1', active=True, user=USER1)
ACCOUNT1_JSON = ACCOUNT1.dumps(serializer='json')
ACCOUNT1_UNBLESSED = ACCOUNT1.to_representation()
ACCOUNT1_UNBLESSED.pop('__faust')
ACCOUNT1_UNBLESSED_JSON = json.dumps(ACCOUNT1_UNBLESSED)
ACCOUNT1_EXTRA_FIELDS = ACCOUNT1.to_representation()
ACCOUNT1_EXTRA_FIELDS.update(join_date='12321321312', foo={'a': 'A', 'b': 'B'}, bar=[1, 2, 3, 4])
ACCOUNT1_EXTRA_FIELDS_JSON = json.dumps(ACCOUNT1_EXTRA_FIELDS)
USER2 = User('B2', 'Elaine', 'Benes')
ACCOUNT2 = Account(id='B1', active=True, user=USER2)
ACCOUNT2_JSON = ACCOUNT2.dumps(serializer='json')
NONFAUST = {'str': 'str', 'int': 1, 'float': 0.3, 'lst': [1, 2, 3], 'dct': {'a': 1, 'b': 2}}
A_BYTE_STR = b'the quick brown fox'
A_STR_STR = 'the quick brown fox'
VALUE_TESTS: typing.List[Case] = [
    Case(ACCOUNT1_JSON, None, 'json', ACCOUNT1),
    Case(ACCOUNT1_EXTRA_FIELDS_JSON, None, 'json', ACCOUNT1),
    Case(ACCOUNT2_JSON, None, 'json', ACCOUNT2),
    Case(ACCOUNT1_UNBLESSED_JSON, Account, 'json', ACCOUNT1),
    Case(json.dumps(NONFAUST), None, 'json', NONFAUST),
    Case(json.dumps(A_STR_STR), None, 'json', A_STR_STR),
    Case(A_STR_STR, None, 'raw', A_BYTE_STR),
    Case(A_BYTE_STR, None, 'raw', A_BYTE_STR),
    Case(json.dumps(A_STR_STR), bytes, 'json', A_BYTE_STR),
    Case(json.dumps(A_STR_STR), str, 'json', A_STR_STR),
    Case(A_BYTE_STR, bytes, 'raw', A_BYTE_STR),
    Case(A_STR_STR, str, 'raw', A_STR_STR),
    Case(ACCOUNT1_JSON, Account, 'json', ACCOUNT1),
    Case(ACCOUNT1_EXTRA_FIELDS_JSON, Account, 'json', ACCOUNT1),
    Case(ACCOUNT1_UNBLESSED_JSON, Account, 'json', ACCOUNT1),
    Case(USER1.dumps(serializer='json'), None, 'json', USER1),
    Case(None, None, 'json', None)
]

@pytest.mark.parametrize('payload,typ,serializer,expected', VALUE_TESTS)
def test_loads_key(payload: typing.Any, typ: typing.Optional[typing.Type], serializer: str, expected: typing.Any, *, app: faust.App) -> None:
    assert app.serializers.loads_key(typ, payload, serializer=serializer) == expected

def test_loads_key__expected_model_received_None(*, app: faust.App) -> None:
    with pytest.raises(KeyDecodeError):
        app.serializers.loads_key(Account, None, serializer='json')

def test_loads_key__propagates_MemoryError(*, app: faust.App) -> None:
    app.serializers._loads = Mock(name='_loads')
    app.serializers._loads.side_effect = MemoryError()
    with pytest.raises(MemoryError):
        app.serializers.loads_key(Account, ACCOUNT1_JSON, serializer='json')

def test_loads_value__propagates_MemoryError(*, app: faust.App) -> None:
    app.serializers._loads = Mock(name='_loads')
    app.serializers._loads.side_effect = MemoryError()
    with pytest.raises(MemoryError):
        app.serializers.loads_value(Account, ACCOUNT1_JSON, serializer='json')

def test_loads_value__expected_model_received_None(*, app: faust.App) -> None:
    with pytest.raises(ValueDecodeError):
        app.serializers.loads_value(Account, None, serializer='json')

@pytest.mark.parametrize('payload,typ,serializer,expected', VALUE_TESTS)
def test_loads_value(payload: typing.Any, typ: typing.Optional[typing.Type], serializer: str, expected: typing.Any, *, app: faust.App) -> None:
    assert app.serializers.loads_value(typ, payload, serializer=serializer) == expected

def test_loads_value_missing_key_raises_error(*, app: faust.App) -> None:
    account = ACCOUNT1.to_representation()
    account.pop('active')
    with pytest.raises(ValueDecodeError):
        app.serializers.loads_value(Account, json.dumps(account), serializer='json')

def test_loads_key_missing_key_raises_error(*, app: faust.App) -> None:
    account = ACCOUNT1.to_representation()
    account.pop('active')
    with pytest.raises(KeyDecodeError):
        app.serializers.loads_key(Account, json.dumps(account), serializer='json')

def test_dumps_value__bytes(*, app: faust.App) -> None:
    assert app.serializers.dumps_value(bytes, b'foo', serializer='json') == b'foo'

@pytest.mark.parametrize('typ,alt,expected', [(str, (), 'raw'), (bytes, (), 'raw'), (str, ('json',), 'json'), (bytes, ('json',), 'json'), (None, (), None)])
def test_serializer_type(typ: typing.Optional[typing.Type], alt: typing.Tuple[str, ...], expected: typing.Optional[str], *, app: faust.App) -> None:
    assert app.serializers._serializer(typ, *alt) == expected

@pytest.mark.parametrize('typ,value,expected_value', [(int, '23', 23), (float, '23.32', 23.32), (Decimal, '23.32', Decimal(23.32)), (str, 'foo', 'foo'), (bytes, 'foo', b'foo')])
def test_prepare_payload(typ: typing.Type, value: str, expected_value: typing.Any, *, app: faust.App) -> None:
    assert app.serializers._prepare_payload(typ, value) == expected_value
