import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest
import pandas as pd
from pandas import Index, Timedelta, merge_asof, option_context, to_datetime
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
from pandas.util._test_decorators import td

class TestAsOfMerge:

    def prep_data(self, df: pd.DataFrame, dedupe: bool = False) -> pd.DataFrame:
        if dedupe:
            df = df.drop_duplicates(['time', 'ticker'], keep='last').reset_index(drop=True)
        df.time = to_datetime(df.time)
        return df

    @pytest.fixture
    def trades(self) -> pd.DataFrame:
        df = pd.DataFrame([['20160525 13:30:00.023', 'MSFT', '51.9500', '75', 'NASDAQ'], ['20160525 13:30:00.038', 'MSFT', '51.9500', '155', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.7700', '100', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9200', '100', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9300', '200', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9300', '300', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9300', '600', 'NASDAQ'], ['20160525 13:30:00.048', 'GOOG', '720.9300', '44', 'NASDAQ'], ['20160525 13:30:00.074', 'AAPL', '98.6700', '478343', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6700', '478343', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6600', '6', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '30', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '75', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '20', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '35', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.6500', '10', 'NASDAQ'], ['20160525 13:30:00.075', 'AAPL', '98.5500', '6', 'ARCA'], ['20160525 13:30:00.075', 'AAPL', '98.5500', '6', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '1000', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '200', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '300', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '400', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '600', 'ARCA'], ['20160525 13:30:00.076', 'AAPL', '98.5600', '200', 'ARCA'], ['20160525 13:30:00.078', 'MSFT', '51.9500', '783', 'NASDAQ'], ['20160525 13:30:00.078', 'MSFT', '51.9500', '100', 'NASDAQ'], ['20160525 13:30:00.078', 'MSFT', '51.9500', '100', 'NASDAQ']], columns='time,ticker,price,quantity,marketCenter'.split(','))
        df['price'] = df['price'].astype('float64')
        df['quantity'] = df['quantity'].astype('int64')
        return self.prep_data(df)

    @pytest.fixture
    def quotes(self) -> pd.DataFrame:
        df = pd.DataFrame([['20160525 13:30:00.023', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.023', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.041', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.50', '720.93'], ['20160525 13:30:00.072', 'GOOG', '720.50', '720.88'], ['20160525 13:30:00.075', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.55', '98.56'], ['20160525 13:30:00.078', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.92', '51.95']], columns='time,ticker,bid,ask'.split(','))
        df['bid'] = df['bid'].astype('float64')
        df['ask'] = df['ask'].astype('float64')
        return self.prep_data(df, dedupe=True)

    @pytest.fixture
    def asof(self) -> pd.DataFrame:
        df = pd.DataFrame([['20160525 13:30:00.023', 'MSFT', '51.95', '75', 'NASDAQ', '51.95', '51.95'], ['20160525 13:30:00.038', 'MSFT', '51.95', '155', 'NASDAQ', '51.95', '51.95'], ['20160525 13:30:00.048', 'GOOG', '720.77', '100', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.92', '100', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '200', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '300', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '600', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '44', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.074', 'AAPL', '98.67', '478343', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.67', '478343', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.66', '6', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '30', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '75', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '20', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '35', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '10', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.55', '6', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.55', '6', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '1000', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '200', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '300', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '400', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '600', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '200', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.078', 'MSFT', '51.95', '783', 'NASDAQ', '51.92', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '100', 'NASDAQ', '51.92', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '100', 'NASDAQ', '51.92', '51.95']], columns='time,ticker,price,quantity,marketCenter,bid,ask'.split(','))
        df['price'] = df['price'].astype('float64')
        df['quantity'] = df['quantity'].astype('int64')
        df['bid'] = df['bid'].astype('float64')
        df['ask'] = df['ask'].astype('float64')
        return self.prep_data(df)

    @pytest.fixture
    def tolerance(self) -> pd.DataFrame:
        df = pd.DataFrame([['20160525 13:30:00.023', 'MSFT', '51.95', '75', 'NASDAQ', '51.95', '51.95'], ['20160525 13:30:00.038', 'MSFT', '51.95', '155', 'NASDAQ', '51.95', '51.95'], ['20160525 13:30:00.048', 'GOOG', '720.77', '100', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.92', '100', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '200', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '300', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '600', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '44', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.074', 'AAPL', '98.67', '478343', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.67', '478343', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.66', '6', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '30', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '75', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '20', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '35', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.65', '10', 'NASDAQ', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.55', '6', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.075', 'AAPL', '98.55', '6', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '1000', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '200', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '300', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '400', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '600', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '200', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.078', 'MSFT', '51.95', '783', 'NASDAQ', '51.92', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '100', 'NASDAQ', '51.92', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '100', 'NASDAQ', '51.92', '51.95']], columns='time,ticker,price,quantity,marketCenter,bid,ask'.split(','))
        df['price'] = df['price'].astype('float64')
        df['quantity'] = df['quantity'].astype('int64')
        df['bid'] = df['bid'].astype('float64')
        df['ask'] = df['ask'].astype('float64')
        return self.prep_data(df)

    def test_examples1(self) -> None:
        """doc-string examples"""
        left = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
        right = pd.DataFrame({'a': [1, 2, 3, 6, 7], 'right_val': [1, 2, 3, 6, 7]})
        expected = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c'], 'right_val': [1, 3, 7]})
        result = merge_asof(left, right, on='a')
        tm.assert_frame_equal(result, expected)

    def test_examples2(self, unit: str) -> None:
        """doc-string examples"""
        if unit == 's':
            pytest.skip("This test is invalid for unit='s' because that would round the trades['time']]")
        trades = pd.DataFrame({'time': to_datetime(['20160525 13:30:00.023', '20160525 13:30:00.038', '201605