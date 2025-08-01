#!/usr/bin/env python3
"""Tests for ExtensionDtype Table Schema integration."""
from collections import OrderedDict
import datetime as dt
import decimal
from io import StringIO
import json
from typing import Any, Callable, OrderedDict as OrderedDictType

import pytest
from pandas import NA, DataFrame, Index, array, read_json
import pandas._testing as tm
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.series import Series
from pandas.tests.extension.date import DateArray, DateDtype
from pandas.tests.extension.decimal.array import DecimalArray, DecimalDtype
from pandas.io.json._table_schema import as_json_table_type, build_table_schema


class TestBuildSchema:
    def test_build_table_schema(self) -> None:
        df: DataFrame = DataFrame({
            'A': DateArray([dt.date(2021, 10, 10)]),
            'B': DecimalArray([decimal.Decimal(10)]),
            'C': array(['pandas'], dtype='string'),
            'D': array([10], dtype='Int64')
        })
        result: dict[str, Any] = build_table_schema(df, version=False)
        expected: dict[str, Any] = {
            'fields': [
                {'name': 'index', 'type': 'integer'},
                {'name': 'A', 'type': 'any', 'extDtype': 'DateDtype'},
                {'name': 'B', 'type': 'number', 'extDtype': 'decimal'},
                {'name': 'C', 'type': 'any', 'extDtype': 'string'},
                {'name': 'D', 'type': 'integer', 'extDtype': 'Int64'}
            ],
            'primaryKey': ['index']
        }
        assert result == expected
        result = build_table_schema(df)
        assert 'pandas_version' in result


class TestTableSchemaType:
    @pytest.mark.parametrize('box', [lambda x: x, Series])
    def test_as_json_table_type_ext_date_array_dtype(self, box: Callable[[Any], Any]) -> None:
        date_data: Any = box(DateArray([dt.date(2021, 10, 10)]))
        assert as_json_table_type(date_data.dtype) == 'any'

    def test_as_json_table_type_ext_date_dtype(self) -> None:
        assert as_json_table_type(DateDtype()) == 'any'

    @pytest.mark.parametrize('box', [lambda x: x, Series])
    def test_as_json_table_type_ext_decimal_array_dtype(self, box: Callable[[Any], Any]) -> None:
        decimal_data: Any = box(DecimalArray([decimal.Decimal(10)]))
        assert as_json_table_type(decimal_data.dtype) == 'number'

    def test_as_json_table_type_ext_decimal_dtype(self) -> None:
        assert as_json_table_type(DecimalDtype()) == 'number'

    @pytest.mark.parametrize('box', [lambda x: x, Series])
    def test_as_json_table_type_ext_string_array_dtype(self, box: Callable[[Any], Any]) -> None:
        string_data: Any = box(array(['pandas'], dtype='string'))
        assert as_json_table_type(string_data.dtype) == 'any'

    def test_as_json_table_type_ext_string_dtype(self) -> None:
        assert as_json_table_type(StringDtype()) == 'any'

    @pytest.mark.parametrize('box', [lambda x: x, Series])
    def test_as_json_table_type_ext_integer_array_dtype(self, box: Callable[[Any], Any]) -> None:
        integer_data: Any = box(array([10], dtype='Int64'))
        assert as_json_table_type(integer_data.dtype) == 'integer'

    def test_as_json_table_type_ext_integer_dtype(self) -> None:
        assert as_json_table_type(Int64Dtype()) == 'integer'


class TestTableOrient:
    @pytest.fixture
    def da(self) -> DateArray:
        """Fixture for creating a DateArray."""
        return DateArray([dt.date(2021, 10, 10)])

    @pytest.fixture
    def dc(self) -> DecimalArray:
        """Fixture for creating a DecimalArray."""
        return DecimalArray([decimal.Decimal(10)])

    @pytest.fixture
    def sa(self) -> Any:
        """Fixture for creating a StringDtype array."""
        return array(['pandas'], dtype='string')

    @pytest.fixture
    def ia(self) -> Any:
        """Fixture for creating an Int64Dtype array."""
        return array([10], dtype='Int64')

    def test_build_date_series(self, da: DateArray) -> None:
        s: Series = Series(da, name='a')
        s.index.name = 'id'
        result: str = s.to_json(orient='table', date_format='iso')
        result_dict: OrderedDictType[str, Any] = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result_dict['schema']
        result_dict['schema'].pop('pandas_version')
        fields: list[dict[str, Any]] = [
            {'name': 'id', 'type': 'integer'},
            {'name': 'a', 'type': 'any', 'extDtype': 'DateDtype'}
        ]
        schema: dict[str, Any] = {'fields': fields, 'primaryKey': ['id']}
        expected: OrderedDictType[str, Any] = OrderedDict([
            ('schema', schema),
            ('data', [OrderedDict([('id', 0), ('a', '2021-10-10T00:00:00.000')])])
        ])
        assert result_dict == expected

    def test_build_decimal_series(self, dc: DecimalArray) -> None:
        s: Series = Series(dc, name='a')
        s.index.name = 'id'
        result: str = s.to_json(orient='table', date_format='iso')
        result_dict: OrderedDictType[str, Any] = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result_dict['schema']
        result_dict['schema'].pop('pandas_version')
        fields: list[dict[str, Any]] = [
            {'name': 'id', 'type': 'integer'},
            {'name': 'a', 'type': 'number', 'extDtype': 'decimal'}
        ]
        schema: dict[str, Any] = {'fields': fields, 'primaryKey': ['id']}
        expected: OrderedDictType[str, Any] = OrderedDict([
            ('schema', schema),
            ('data', [OrderedDict([('id', 0), ('a', '10')])])
        ])
        assert result_dict == expected

    def test_build_string_series(self, sa: Any) -> None:
        s: Series = Series(sa, name='a')
        s.index.name = 'id'
        result: str = s.to_json(orient='table', date_format='iso')
        result_dict: OrderedDictType[str, Any] = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result_dict['schema']
        result_dict['schema'].pop('pandas_version')
        fields: list[dict[str, Any]] = [
            {'name': 'id', 'type': 'integer'},
            {'name': 'a', 'type': 'any', 'extDtype': 'string'}
        ]
        schema: dict[str, Any] = {'fields': fields, 'primaryKey': ['id']}
        expected: OrderedDictType[str, Any] = OrderedDict([
            ('schema', schema),
            ('data', [OrderedDict([('id', 0), ('a', 'pandas')])])
        ])
        assert result_dict == expected

    def test_build_int64_series(self, ia: Any) -> None:
        s: Series = Series(ia, name='a')
        s.index.name = 'id'
        result: str = s.to_json(orient='table', date_format='iso')
        result_dict: OrderedDictType[str, Any] = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result_dict['schema']
        result_dict['schema'].pop('pandas_version')
        fields: list[dict[str, Any]] = [
            {'name': 'id', 'type': 'integer'},
            {'name': 'a', 'type': 'integer', 'extDtype': 'Int64'}
        ]
        schema: dict[str, Any] = {'fields': fields, 'primaryKey': ['id']}
        expected: OrderedDictType[str, Any] = OrderedDict([
            ('schema', schema),
            ('data', [OrderedDict([('id', 0), ('a', 10)])])
        ])
        assert result_dict == expected

    def test_to_json(self, da: DateArray, dc: DecimalArray, sa: Any, ia: Any) -> None:
        df: DataFrame = DataFrame({
            'A': da,
            'B': dc,
            'C': sa,
            'D': ia
        })
        df.index.name = 'idx'
        result: str = df.to_json(orient='table', date_format='iso')
        result_dict: OrderedDictType[str, Any] = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result_dict['schema']
        result_dict['schema'].pop('pandas_version')
        fields: list[OrderedDictType[str, Any]] = [
            OrderedDict({'name': 'idx', 'type': 'integer'}),
            OrderedDict({'name': 'A', 'type': 'any', 'extDtype': 'DateDtype'}),
            OrderedDict({'name': 'B', 'type': 'number', 'extDtype': 'decimal'}),
            OrderedDict({'name': 'C', 'type': 'any', 'extDtype': 'string'}),
            OrderedDict({'name': 'D', 'type': 'integer', 'extDtype': 'Int64'})
        ]
        schema: OrderedDictType[str, Any] = OrderedDict({'fields': fields, 'primaryKey': ['idx']})
        data: list[OrderedDictType[str, Any]] = [
            OrderedDict([('idx', 0), ('A', '2021-10-10T00:00:00.000'), ('B', '10'), ('C', 'pandas'), ('D', 10)])
        ]
        expected: OrderedDictType[str, Any] = OrderedDict([
            ('schema', schema),
            ('data', data)
        ])
        assert result_dict == expected

    def test_json_ext_dtype_reading_roundtrip(self) -> None:
        df: DataFrame = DataFrame({
            'a': Series([2, NA], dtype='Int64'),
            'b': Series([1.5, NA], dtype='Float64'),
            'c': Series([True, NA], dtype='boolean')
        }, index=Index([1, NA], dtype='Int64'))
        expected: DataFrame = df.copy()
        data_json: str = df.to_json(orient='table', indent=4)
        result: DataFrame = read_json(StringIO(data_json), orient='table')
        tm.assert_frame_equal(result, expected)

    def test_json_ext_dtype_reading(self) -> None:
        data_json: str = (
            '{\n'
            '    "schema":{\n'
            '        "fields":[\n'
            '            {\n'
            '                "name":"a",\n'
            '                "type":"integer",\n'
            '                "extDtype":"Int64"\n'
            '            }\n'
            '        ],\n'
            '    },\n'
            '    "data":[\n'
            '        {\n'
            '            "a":2\n'
            '        },\n'
            '        {\n'
            '            "a":null\n'
            '        }\n'
            '    ]\n'
            '}'
        )
        result: DataFrame = read_json(StringIO(data_json), orient='table')
        expected: DataFrame = DataFrame({'a': Series([2, NA], dtype='Int64')})
        tm.assert_frame_equal(result, expected)
