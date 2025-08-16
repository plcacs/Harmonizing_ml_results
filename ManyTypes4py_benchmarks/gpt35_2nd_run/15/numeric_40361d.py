    def increment(self, accumulator: t.Optional[str] = None) -> int:
    def float_number(self, start: float = -1000.0, end: float = 1000.0, precision: int = 15) -> float:
    def floats(self, start: float = 0, end: float = 1, n: int = 10, precision: int = 15) -> t.List[float]:
    def integer_number(self, start: int = -1000, end: int = 1000) -> int:
    def integers(self, start: int = 0, end: int = 10, n: int = 10) -> t.List[int]:
    def complex_number(self, start_real: float = 0.0, end_real: float = 1.0, start_imag: float = 0.0, end_imag: float = 1.0, precision_real: int = 15, precision_imag: int = 15) -> complex:
    def complexes(self, start_real: float = 0, end_real: float = 1, start_imag: float = 0, end_imag: float = 1, precision_real: int = 15, precision_imag: int = 15, n: int = 10) -> t.List[complex]:
    def decimal_number(self, start: float = -1000.0, end: float = 1000.0) -> Decimal:
    def decimals(self, start: float = 0.0, end: float = 1000.0, n: int = 10) -> t.List[Decimal]:
    def matrix(self, m: int = 10, n: int = 10, num_type: NumType = NumType.FLOAT, **kwargs) -> Matrix:
