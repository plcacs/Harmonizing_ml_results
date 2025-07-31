import decimal
import re
import pytest
from typing import List, Union
from mimesis import Numeric
from mimesis.enums import NumType
from mimesis.exceptions import NonEnumerableError
from . import patterns

class TestNumbers:
    @pytest.fixture
    def numeric(self) -> Numeric:
        return Numeric()

    def test_str(self, numeric: Numeric) -> None:
        assert re.match(patterns.PROVIDER_STR_REGEX, str(numeric))

    def test_incremental(self) -> None:
        numeric: Numeric = Numeric()
        for i in range(1, 50 + 1):
            assert numeric.increment() == i

    def test_incremental_with_accumulator(self, numeric: Numeric) -> None:
        for i in range(1, 50):
            for key in ('a', 'b', 'c'):
                assert numeric.increment(accumulator=key) == i

    @pytest.mark.parametrize('start, end', [(1.2, 10), (10.4, 20.0), (20.3, 30.8)])
    def test_floats(self, numeric: Numeric, start: float, end: float) -> None:
        result: List[float] = numeric.floats(start, end)
        assert max(result) <= end
        assert min(result) >= start
        assert len(result) == 10
        assert isinstance(result, list)
        result = numeric.floats(n=1000)
        assert len(result) == 1000
        result = numeric.floats(precision=4)
        for e in result:
            # When e is a float, converting to string may not have a decimal point
            parts = str(e).split('.')
            if len(parts) > 1:
                assert len(parts[1]) <= 4

    @pytest.mark.parametrize('start, end', [(1, 10), (10, 20), (20, 30)])
    def test_integers(self, numeric: Numeric, start: int, end: int) -> None:
        result: List[int] = numeric.integers(start=start, end=end)
        assert max(result) <= end
        assert min(result) >= start
        assert isinstance(result, list)
        element: int = numeric.random.choice(result)
        assert isinstance(element, int)

    @pytest.mark.parametrize('start, end', [(1, 10), (10, 20), (20, 30)])
    def test_decimals(self, numeric: Numeric, start: int, end: int) -> None:
        result: List[decimal.Decimal] = numeric.decimals(start=start, end=end)
        # Convert elements to float for comparison purposes
        assert max(result) <= end
        assert min(result) >= start
        assert isinstance(result, list)
        element: decimal.Decimal = numeric.random.choice(result)
        assert isinstance(element, decimal.Decimal)

    @pytest.mark.parametrize('start_real, end_real, start_imag, end_imag',
                             [(1.2, 10, 1, 2.4), (10.4, 20.0, 2.3, 10), (20.3, 30.8, 2.4, 4.5)])
    def test_complexes(self, numeric: Numeric, start_real: float, end_real: float,
                       start_imag: float, end_imag: float) -> None:
        result: List[complex] = numeric.complexes(start_real, end_real, start_imag, end_imag)
        assert max((e.real for e in result)) <= end_real
        assert min((e.real for e in result)) >= start_real
        assert max((e.imag for e in result)) <= end_imag
        assert min((e.imag for e in result)) >= start_imag
        assert len(result) == 10
        assert isinstance(result, list)
        result = numeric.complexes(n=1000)
        assert len(result) == 1000
        result = numeric.complexes(precision_real=4, precision_imag=6)
        for e in result:
            real_part = str(e.real).split('.')
            imag_part = str(e.imag).split('.')
            if len(real_part) > 1:
                assert len(real_part[1]) <= 4
            if len(imag_part) > 1:
                assert len(imag_part[1]) <= 6

    @pytest.mark.parametrize('sr, er, si, ei, pr, pi',
                             [(1.2, 10, 1, 2.4, 15, 15), (10.4, 20.0, 2.3, 10, 10, 10), (20.3, 30.8, 2.4, 4.5, 12, 12)])
    def test_complex_number(self, numeric: Numeric, sr: float, er: float, si: float,
                            ei: float, pr: int, pi: int) -> None:
        result: complex = numeric.complex_number(start_real=sr, end_real=er,
                                                 start_imag=si, end_imag=ei,
                                                 precision_real=pr, precision_imag=pi)
        assert isinstance(result, complex)
        real_part = str(result.real).split('.')
        imag_part = str(result.imag).split('.')
        if len(real_part) > 1:
            assert len(real_part[1]) <= pr
        if len(imag_part) > 1:
            assert len(imag_part[1]) <= pi

    def test_matrix(self, numeric: Numeric) -> None:
        with pytest.raises(NonEnumerableError):
            numeric.matrix(num_type='int')
        # Testing matrix of floats
        result: List[List[float]] = numeric.matrix(precision=4)
        assert len(result) == 10
        for row in result:
            assert len(row) == 10
            for e in row:
                assert isinstance(e, float)
                parts = str(e).split('.')
                if len(parts) > 1:
                    assert len(parts[1]) <= 4
        # Testing matrix of integers
        result_int: List[List[int]] = numeric.matrix(m=5, n=5, num_type=NumType.INTEGER, start=5)
        assert len(result_int) == 5
        for row in result_int:
            assert len(row) == 5
            assert min(row) >= 5
            for e in row:
                assert isinstance(e, int)
        # Testing matrix of complex numbers
        precision_real: int = 4
        precision_imag: int = 6
        result_complex: List[List[complex]] = numeric.matrix(num_type=NumType.COMPLEX,
                                                              precision_real=precision_real,
                                                              precision_imag=precision_imag)
        # Modify one element to ensure complex formatting verification
        result_complex[0][0] = 0.0001 + 1e-06j
        assert len(result_complex) == 10
        for row in result_complex:
            assert len(row) == 10
            for e in row:
                real_str: str = f'{e.real:.{precision_real}f}'
                imag_str: str = f'{e.imag:.{precision_imag}f}'
                assert float(real_str) == e.real
                assert float(imag_str) == e.imag
                if '.' in real_str:
                    assert len(real_str.split('.')[1]) <= precision_real
                if '.' in imag_str:
                    assert len(imag_str.split('.')[1]) <= precision_imag

    def test_integer(self, numeric: Numeric) -> None:
        result: int = numeric.integer_number(-100, 100)
        assert isinstance(result, int)
        assert -100 <= result <= 100

    def test_float(self, numeric: Numeric) -> None:
        result: float = numeric.float_number(-100, 100, precision=15)
        assert isinstance(result, float)
        assert -100 <= result <= 100
        parts = str(result).split('.')
        if len(parts) > 1:
            assert len(parts[1]) <= 15

    def test_decimal(self, numeric: Numeric) -> None:
        result: decimal.Decimal = numeric.decimal_number(-100, 100)
        assert -100 <= result <= 100
        assert isinstance(result, decimal.Decimal)

class TestSeededNumbers:
    @pytest.fixture
    def n1(self, seed: int) -> Numeric:
        return Numeric(seed=seed)

    @pytest.fixture
    def n2(self, seed: int) -> Numeric:
        return Numeric(seed=seed)

    def test_incremental(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.increment() == n2.increment()

    def test_floats(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.floats() == n2.floats()
        assert n1.floats(n=5) == n2.floats(n=5)

    def test_decimals(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.decimals() == n2.decimals()
        assert n1.decimals(n=5) == n2.decimals(n=5)

    def test_integers(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.integers() == n2.integers()
        assert n1.integers(start=-999, end=999, n=10) == n2.integers(start=-999, end=999, n=10)

    def test_complexes(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.complexes() == n2.complexes()
        assert n1.complexes(n=5) == n2.complexes(n=5)

    def test_matrix(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.matrix() == n2.matrix()
        assert n1.matrix(n=5) == n2.matrix(n=5)

    def test_integer(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.integer_number() == n2.integer_number()

    def test_float(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.float_number() == n2.float_number()

    def test_decimal(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.decimal_number() == n2.decimal_number()

    def test_complex_number(self, n1: Numeric, n2: Numeric) -> None:
        assert n1.complex_number() == n2.complex_number()