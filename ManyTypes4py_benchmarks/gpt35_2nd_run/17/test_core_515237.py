    def __init__(self, name: str, text_field: str, lower_field: str, words_field: str) -> None:
    def run(self, text: str) -> Optional[FieldMap]:
    def run(self, text: str) -> Optional[FieldMap]:
    def run(self, text: str) -> Optional[FieldMap]:
    def run(self, text: str, *args: Any) -> Optional[FieldMap]:
    def run(self, text: str, **kwargs: Any) -> Optional[FieldMap]:
    def run(self, num_squared: int) -> Optional[FieldMap]:
    def run(self, double_num_squared: int) -> Optional[FieldMap]:
def square(x: DataPoint) -> DataPoint:
def modify_in_place(x: DataPoint) -> DataPoint:
    def _get_x(self, num: int = 8, text: str = 'Henry has fun') -> DataPoint:
    def _get_x_dict(self) -> DataPoint:
    def test_numeric_mapper(self) -> None:
    def test_text_mapper(self) -> None:
    def test_mapper_same_field(self) -> None:
    def test_mapper_default_args(self) -> None:
    def test_mapper_in_place(self) -> None:
    def test_mapper_returns_none(self) -> None:
    def test_mapper_pre(self) -> None:
    def test_mapper_pre_decorator(self) -> None:
    def test_decorator_mapper_memoized(self) -> None:
    def test_decorator_mapper_memoized_none(self) -> None:
    def test_decorator_mapper_memoized_use_memoize_key(self) -> None:
    def test_decorator_mapper_not_memoized(self) -> None:
    def test_mapper_pre_memoized(self) -> None:
    def test_mapper_decorator_no_parens(self) -> None:
    def test_mapper_with_args_kwargs(self) -> None:
    def test_get_hashable_hashable(self) -> None:
    def test_get_hashable_dict(self) -> None:
    def test_get_hashable_list(self) -> None:
    def test_get_hashable_series(self) -> None:
    def test_get_hashable_series_with_doc(self) -> None:
    def test_get_hashable_ndarray(self) -> None:
    def test_get_hashable_unhashable(self) -> None:
