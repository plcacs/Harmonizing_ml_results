def dtype_for_elements_strategy(s: st.SearchStrategy) -> st.SearchStrategy:
def infer_dtype_if_necessary(dtype: Optional[Any], values: Any, elements: st.SearchStrategy, draw: Any) -> Any:
def elements_and_dtype(elements: Optional[st.SearchStrategy], dtype: Optional[Any], source: Optional[str]) -> Tuple[st.SearchStrategy, Any]:
class ValueIndexStrategy(st.SearchStrategy):
def do_draw(self, data: Any) -> Any:
def range_indexes(min_size: int = 0, max_size: Optional[int] = None, name: Optional[st.SearchStrategy] = st.none()) -> st.SearchStrategy:
def indexes(*, elements: Optional[st.SearchStrategy] = None, dtype: Optional[Any] = None, min_size: int = 0, max_size: Optional[int] = None, unique: bool = True, name: Optional[st.SearchStrategy] = st.none()) -> ValueIndexStrategy:
def series(*, elements: Optional[st.SearchStrategy] = None, dtype: Optional[Any] = None, index: Optional[st.SearchStrategy] = None, fill: Optional[Any] = None, unique: bool = False, name: Optional[st.SearchStrategy] = st.none()) -> st.SearchStrategy:
@attr.s(slots=True)
class column(Generic[Ex]):
def columns(names_or_number, *, dtype: Optional[Any] = None, elements: Optional[st.SearchStrategy] = None, fill: Optional[Any] = None, unique: bool = False) -> List[column]:
def data_frames(columns: Optional[Union[Iterable[column], int, float]] = None, *, rows: Optional[st.SearchStrategy] = None, index: Optional[st.SearchStrategy] = None) -> st.SearchStrategy:
