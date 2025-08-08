def _align_core_single_unary_op(term: F) -> tuple[F, dict[str, Index]]:
    axes: dict[str, Index] = None

def _zip_axes_from_type(typ: type, new_axes: Sequence[Index]) -> dict[str, Index]:

def _any_pandas_objects(terms: Sequence[F]) -> bool:

def _filter_special_cases(f: Callable) -> Callable:

def _align_core(terms: Sequence[F]) -> tuple[type, dict[str, Index]]:

def align_terms(terms: Sequence[F]) -> tuple[type, dict[str, Index], str]:

def reconstruct_object(typ: type, obj: object, axes: dict[str, Index], dtype: type, name: str) -> object:
