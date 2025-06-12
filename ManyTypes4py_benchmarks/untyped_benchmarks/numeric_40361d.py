"""Provides data related to numbers."""
import typing as t
from collections import defaultdict
from decimal import Decimal
from mimesis.enums import NumType
from mimesis.providers.base import BaseProvider
from mimesis.types import Matrix
__all__ = ['Numeric']

class Numeric(BaseProvider):
    """A provider for generating numeric data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__increment_dict = defaultdict(int)
        self.__default_accumulator_value = 'default'

    class Meta:
        name = 'numeric'

    def increment(self, accumulator=None):
        """Generates an incrementing number.

        Each call of this method returns an incrementing number (with the step of +1).

        If **accumulator** passed then increments number associated with it.

        Example:
            >>> self.increment()
            1
            >>> self.increment(accumulator="a")
            1
            >>> self.increment()
            2
            >>> self.increment(accumulator="a")
            2
            >>> self.increment(accumulator="b")
            1
            >>> self.increment(accumulator="a")
            3

        :param accumulator: Accumulator (used to create associative incrementation).
        :return: Integer.
        """
        if not accumulator:
            accumulator = self.__default_accumulator_value
        self.__increment_dict[accumulator] += 1
        return self.__increment_dict[accumulator]

    def float_number(self, start=-1000.0, end=1000.0, precision=15):
        """Generates a random float number in range [start, end].

        :param start: Start range.
        :param end:  End range.
        :param precision: Round a number to a given
            precision in decimal digits, default is 15.
        :return: Float.
        """
        return self.random.uniform(start, end, precision)

    def floats(self, start=0, end=1, n=10, precision=15):
        """Generates a list of random float numbers.

        :param start: Start range.
        :param end: End range.
        :param n: Length of the list.
        :param precision: Round a number to a given
            precision in decimal digits, default is 15.
        :return: The list of floating-point numbers.
        """
        return [self.float_number(start, end, precision) for _ in range(n)]

    def integer_number(self, start=-1000, end=1000):
        """Generates a random integer from start to end.

        :param start: Start range.
        :param end: End range.
        :return: Integer.
        """
        return self.random.randint(start, end)

    def integers(self, start=0, end=10, n=10):
        """Generates a list of random integers.

        :param start: Start.
        :param end: End.
        :param n: Length of the list.
        :return: List of integers.

        :Example:
            [-20, -19, -18, -17]
        """
        return self.random.randints(n, start, end)

    def complex_number(self, start_real=0.0, end_real=1.0, start_imag=0.0, end_imag=1.0, precision_real=15, precision_imag=15):
        """Generates a random complex number.

        :param start_real: Start real range.
        :param end_real: End real range.
        :param start_imag: Start imaginary range.
        :param end_imag: End imaginary range.
        :param precision_real:  Round a real part of
            number to a given precision.
        :param precision_imag:  Round the imaginary part of
            number to a given precision.
        :return: Complex numbers.
        """
        real_part = self.random.uniform(start_real, end_real, precision_real)
        imag_part = self.random.uniform(start_imag, end_imag, precision_imag)
        return complex(real_part, imag_part)

    def complexes(self, start_real=0, end_real=1, start_imag=0, end_imag=1, precision_real=15, precision_imag=15, n=10):
        """Generates a list of random complex numbers.

        :param start_real: Start real range.
        :param end_real: End real range.
        :param start_imag: Start imaginary range.
        :param end_imag: End imaginary range.
        :param precision_real:  Round a real part of
            number to a given precision.
        :param precision_imag:  Round the imaginary part of
            number to a given precision.
        :param n: Length of the list.
        :return: A list of random complex numbers.
        """
        numbers = []
        for _ in range(n):
            numbers.append(self.complex_number(start_real=start_real, end_real=end_real, start_imag=start_imag, end_imag=end_imag, precision_real=precision_real, precision_imag=precision_imag))
        return numbers

    def decimal_number(self, start=-1000.0, end=1000.0):
        """Generates a random decimal number.

        :param start:  Start range.
        :param end: End range.
        :return: :py:class:`decimal.Decimal` object.
        """
        return Decimal.from_float(self.float_number(start, end))

    def decimals(self, start=0.0, end=1000.0, n=10):
        """Generates a list of decimal numbers.

        :param start: Start range.
        :param end: End range.
        :param n: Length of the list.
        :return: A list of :py:class:`decimal.Decimal` objects.
        """
        return [self.decimal_number(start, end) for _ in range(n)]

    def matrix(self, m=10, n=10, num_type=NumType.FLOAT, **kwargs):
        """Generates m x n matrix with a random numbers.

        This method works with a variety of types,
        so you can pass method-specific `**kwargs`.

        :param m: Number of rows.
        :param n: Number of columns.
        :param num_type: NumType enum object.
        :param kwargs: Other method-specific arguments.
        :return: A matrix of random numbers.
        """
        key = self.validate_enum(num_type, NumType)
        kwargs.update({'n': n})
        method = getattr(self, key)
        return [method(**kwargs) for _ in range(m)]