from decimal import Decimal
import pytest
from mode.utils.mocks import Mock
from faust import Record
from faust.exceptions import ValidationError
from faust.models.fields import BooleanField, BytesField, DecimalField, FieldDescriptor

class X(Record):
    pass

class test_ValidationError:

    @pytest.fixture()
    def field(self) -> DecimalField:
        return DecimalField(model=X, field='foo')

    @pytest.fixture()
    def error(self, *, field: Union[str, pydantic.fields.ModelField]) -> ValidationError:
        return ValidationError('error', field=field)

    def test_repr(self, *, error: Union[Exception, dict, werkzeug.exceptions.Unauthorized]) -> None:
        assert repr(error)

    def test_str(self, *, error: Union[str, None, KeyError]) -> None:
        assert str(error)

class test_FieldDescriptor:

    def test_validate(self) -> None:
        f = FieldDescriptor()
        assert list(f.validate('foo')) == []

class test_BooleanField:

    @pytest.fixture()
    def model(self) -> Mock:
        model = Mock(name='model')
        model.__name__ = 'Model'
        return model

    @pytest.fixture()
    def field(self, *, model) -> DecimalField:
        return self._new_field(model, required=True)

    def _new_field(self, model: Union[bool, dict[str, typing.Any], None], required: Union[bool, dict[str, typing.Any], None], **kwargs) -> BooleanField:
        return BooleanField(field='foo', type=bool, required=True, model=model, coerce=True, **kwargs)

    @pytest.mark.parametrize('value', [True, False])
    def test_validate_bool(self, value: Union[typing.Iterable[T], application.domain.models.T, typing.Type], *, field: Union[typing.Iterable[T], application.domain.models.T, typing.Type]) -> None:
        assert not list(field.validate(value))

    @pytest.mark.parametrize('value', ['', None, 12, 3.2, object])
    def test_validate_other(self, value: Union[list, typing.Type], *, field: Union[list, typing.Type]) -> None:
        errors = list(field.validate(value))
        assert errors
        assert str(errors[0]).startswith('foo must be True or False, of type bool')

    @pytest.mark.parametrize('value,expected', [('', False), ('foo', True), (0, False), (1, True), (999, True), (object(), True), (None, False), ({}, False), ([], False), ([1], True)])
    def test_prepare_value__when_coerce(self, value: Union[str, list[str], dict], expected: Union[str, list[str], dict], *, field: Union[str, list[str], dict]) -> None:
        assert field.prepare_value(value) is expected

    def test_prepare_value__no_coerce(self, *, field: Union[str, None, typing.Any]) -> None:
        assert field.prepare_value(None, coerce=False) is None

class test_DecimalField:

    def test_init_options(self) -> None:
        assert DecimalField(max_digits=3).max_digits == 3
        assert DecimalField(max_decimal_places=4).max_decimal_places == 4
        f = DecimalField(max_digits=3, max_decimal_places=4)
        f2 = f.clone()
        assert f2.max_digits == 3
        assert f2.max_decimal_places == 4
        f3 = DecimalField()
        assert f3.max_digits is None
        assert f3.max_decimal_places is None
        f4 = f3.clone()
        assert f4.max_digits is None
        assert f4.max_decimal_places is None

    @pytest.mark.parametrize('value', [Decimal('Inf'), Decimal('NaN'), Decimal('sNaN')])
    def test_infinite(self, value: Union[dict[str, typing.Any], float, T]) -> None:
        f = DecimalField(coerce=True, field='foo')
        with pytest.raises(ValidationError):
            raise next(f.validate(value))

    @pytest.mark.parametrize('value,places,digits', [(Decimal(4.1), 100, 2), (Decimal(4.1), 100, 2), (Decimal(4.1), None, 2), (Decimal(4.12), 100, None), (Decimal(4.123), 100, None), (4.1234, 100, 2), (Decimal(4.1234), 100, 2), (Decimal(123456612341.1234), 100, 100)])
    def test_max_decimal_places__good(self, value: Union[str, tuple[int], dict[str, typing.Union[str,None]]], places: Union[float, int, utils.reporting.DefaultStat], digits: Union[float, int, utils.reporting.DefaultStat]) -> None:
        f = DecimalField(max_decimal_places=places, max_digits=digits, coerce=True, field='foo')
        d = f.prepare_value(value)
        for error in f.validate(d):
            raise error

    @pytest.mark.parametrize('value', [Decimal(1.12412421421), Decimal(1.12345), Decimal(123456788.12345)])
    def test_max_decimal_places__bad(self, value: Union[dict[raiden.utils.ChannelID, float], T, typing.Callable, None]) -> None:
        f = DecimalField(max_decimal_places=4, coerce=True, field='foo')
        with pytest.raises(ValidationError):
            raise next(f.validate(value))

    @pytest.mark.parametrize('value', [Decimal(12345.12412421421), Decimal(123456.12345), Decimal(123456788.12345)])
    def test_max_digits__bad(self, value: Union[T, dict[raiden.utils.ChannelID, float], typing.Callable, None]) -> None:
        f = DecimalField(max_digits=4, coerce=True, field='foo')
        with pytest.raises(ValidationError):
            raise next(f.validate(value))

class test_BytesField:

    def test_init_options(self) -> None:
        assert BytesField(encoding='latin1').encoding == 'latin1'
        assert BytesField(errors='replace').errors == 'replace'
        f = BytesField(encoding='latin1', errors='replace')
        f2 = f.clone()
        assert f2.encoding == 'latin1'
        assert f2.errors == 'replace'
        f3 = BytesField()
        assert f3.encoding == 'utf-8'
        assert f3.errors == 'strict'
        f4 = f3.clone()
        assert f4.encoding == 'utf-8'
        assert f4.errors == 'strict'

    @pytest.mark.parametrize('value,coerce,trim,expected_result', [('foo', True, False, b'foo'), (b'foo', True, False, b'foo'), ('foo', False, False, 'foo'), ('  fo o   ', True, True, b'fo o'), (b'  fo o   ', True, True, b'fo o'), ('  fo o   ', True, False, b'  fo o   ')])
    def test_prepare_value(self, value: Union[bool, float, str], coerce: Union[bool, tuple[typing.Union[bool,...]], None], trim: Union[bool, tuple[typing.Union[bool,...]], None], expected_result: Union[bool, float, str]) -> None:
        f = BytesField(coerce=coerce, trim_whitespace=trim)
        assert f.prepare_value(value) == expected_result