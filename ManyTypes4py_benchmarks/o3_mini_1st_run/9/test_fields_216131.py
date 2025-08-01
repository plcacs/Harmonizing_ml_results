from decimal import Decimal
from typing import Any, Iterator, Optional, Union

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
    def error(self, *, field: DecimalField) -> ValidationError:
        return ValidationError('error', field=field)

    def test_repr(self, *, error: ValidationError) -> None:
        assert repr(error)

    def test_str(self, *, error: ValidationError) -> None:
        assert str(error)


class test_FieldDescriptor:

    def test_validate(self) -> None:
        f: FieldDescriptor = FieldDescriptor()
        assert list(f.validate('foo')) == []


class test_BooleanField:

    @pytest.fixture()
    def model(self) -> Mock:
        model: Mock = Mock(name='model')
        model.__name__ = 'Model'
        return model

    @pytest.fixture()
    def field(self, *, model: Any) -> BooleanField:
        return self._new_field(model, required=True)

    def _new_field(self, model: Any, required: bool, **kwargs: Any) -> BooleanField:
        return BooleanField(field='foo', type=bool, required=True, model=model, coerce=True, **kwargs)

    @pytest.mark.parametrize('value', [True, False])
    def test_validate_bool(self, value: bool, *, field: BooleanField) -> None:
        assert not list(field.validate(value))

    @pytest.mark.parametrize('value', ['', None, 12, 3.2, object])
    def test_validate_other(self, value: Any, *, field: BooleanField) -> None:
        errors = list(field.validate(value))
        assert errors
        assert str(errors[0]).startswith('foo must be True or False, of type bool')

    @pytest.mark.parametrize(
        'value,expected',
        [
            ('', False),
            ('foo', True),
            (0, False),
            (1, True),
            (999, True),
            (object(), True),
            (None, False),
            ({}, False),
            ([], False),
            ([1], True),
        ],
    )
    def test_prepare_value__when_coerce(self, value: Any, expected: bool, *, field: BooleanField) -> None:
        assert field.prepare_value(value) is expected

    def test_prepare_value__no_coerce(self, *, field: BooleanField) -> None:
        assert field.prepare_value(None, coerce=False) is None


class test_DecimalField:

    def test_init_options(self) -> None:
        assert DecimalField(max_digits=3).max_digits == 3
        assert DecimalField(max_decimal_places=4).max_decimal_places == 4
        f: DecimalField = DecimalField(max_digits=3, max_decimal_places=4)
        f2: DecimalField = f.clone()
        assert f2.max_digits == 3
        assert f2.max_decimal_places == 4
        f3: DecimalField = DecimalField()
        assert f3.max_digits is None
        assert f3.max_decimal_places is None
        f4: DecimalField = f3.clone()
        assert f4.max_digits is None
        assert f4.max_decimal_places is None

    @pytest.mark.parametrize('value', [Decimal('Inf'), Decimal('NaN'), Decimal('sNaN')])
    def test_infinite(self, value: Decimal) -> None:
        f: DecimalField = DecimalField(coerce=True, field='foo')
        with pytest.raises(ValidationError):
            raise next(f.validate(value))  # type: ignore

    @pytest.mark.parametrize(
        'value,places,digits',
        [
            (Decimal(4.1), 100, 2),
            (Decimal(4.1), 100, 2),
            (Decimal(4.1), None, 2),
            (Decimal(4.12), 100, None),
            (Decimal(4.123), 100, None),
            (4.1234, 100, 2),
            (Decimal(4.1234), 100, 2),
            (Decimal(123456612341.1234), 100, 100),
        ],
    )
    def test_max_decimal_places__good(
        self, value: Union[Decimal, float], places: Optional[int], digits: Optional[int]
    ) -> None:
        f: DecimalField = DecimalField(max_decimal_places=places, max_digits=digits, coerce=True, field='foo')
        d: Decimal = f.prepare_value(value)
        for error in f.validate(d):  # type: Iterator[ValidationError]
            raise error

    @pytest.mark.parametrize('value', [Decimal(1.12412421421), Decimal(1.12345), Decimal(123456788.12345)])
    def test_max_decimal_places__bad(self, value: Decimal) -> None:
        f: DecimalField = DecimalField(max_decimal_places=4, coerce=True, field='foo')
        with pytest.raises(ValidationError):
            raise next(f.validate(value))  # type: ignore

    @pytest.mark.parametrize('value', [Decimal(12345.12412421421), Decimal(123456.12345), Decimal(123456788.12345)])
    def test_max_digits__bad(self, value: Decimal) -> None:
        f: DecimalField = DecimalField(max_digits=4, coerce=True, field='foo')
        with pytest.raises(ValidationError):
            raise next(f.validate(value))  # type: ignore


class test_BytesField:

    def test_init_options(self) -> None:
        assert BytesField(encoding='latin1').encoding == 'latin1'
        assert BytesField(errors='replace').errors == 'replace'
        f: BytesField = BytesField(encoding='latin1', errors='replace')
        f2: BytesField = f.clone()
        assert f2.encoding == 'latin1'
        assert f2.errors == 'replace'
        f3: BytesField = BytesField()
        assert f3.encoding == 'utf-8'
        assert f3.errors == 'strict'
        f4: BytesField = f3.clone()
        assert f4.encoding == 'utf-8'
        assert f4.errors == 'strict'

    @pytest.mark.parametrize(
        'value,coerce,trim,expected_result',
        [
            ('foo', True, False, b'foo'),
            (b'foo', True, False, b'foo'),
            ('foo', False, False, 'foo'),
            ('  fo o   ', True, True, b'fo o'),
            (b'  fo o   ', True, True, b'fo o'),
            ('  fo o   ', True, False, b'  fo o   '),
        ],
    )
    def test_prepare_value(
        self, value: Union[str, bytes], coerce: bool, trim: bool, expected_result: Union[str, bytes]
    ) -> None:
        f: BytesField = BytesField(coerce=coerce, trim_whitespace=trim)
        assert f.prepare_value(value) == expected_result
