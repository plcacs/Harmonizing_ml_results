def get_parameters(f: Callable, allow_args: bool = False, allow_kwargs: bool = False) -> List[str]:
def is_hashable(obj: Any) -> bool:
def get_hashable(obj: Any) -> Hashable:
class BaseMapper:
    def __init__(self, name: str, pre: List[BaseMapper], memoize: bool, memoize_key: Optional[HashingFunction] = None) -> None:
    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
class Mapper(BaseMapper):
    def __init__(self, name: str, field_names: Optional[Dict[str, str]] = None, mapped_field_names: Optional[Dict[str, str]] = None, pre: Optional[List[BaseMapper]] = None, memoize: bool = False, memoize_key: Optional[HashingFunction] = None) -> None:
    def run(self, **kwargs: Any) -> Optional[FieldMap]:
    def _update_fields(self, x: DataPoint, mapped_fields: FieldMap) -> DataPoint:
    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
class LambdaMapper(BaseMapper):
    def __init__(self, name: str, f: MapFunction, pre: Optional[List[BaseMapper]] = None, memoize: bool = False, memoize_key: Optional[HashingFunction] = None) -> None:
    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
class lambda_mapper:
    def __init__(self, name: Optional[str] = None, pre: Optional[List[BaseMapper]] = None, memoize: bool = False, memoize_key: Optional[HashingFunction] = None) -> None:
    def __call__(self, f: MapFunction) -> LambdaMapper:
