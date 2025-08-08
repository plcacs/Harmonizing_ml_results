def set_use_numexpr(v: bool = True) -> None:
    global USE_NUMEXPR

def set_numexpr_threads(n: int = None) -> None:

def _evaluate_standard(op, op_str, left_op, right_op) -> Any:

def _can_use_numexpr(op, op_str, left_op, right_op, dtype_check) -> bool:

def _evaluate_numexpr(op, op_str, left_op, right_op) -> Any:

def _where_standard(cond, left_op, right_op) -> Any:

def _where_numexpr(cond, left_op, right_op) -> Any:

def _has_bool_dtype(x) -> bool:

def _bool_arith_fallback(op_str, left_op, right_op) -> bool:

def evaluate(op, left_op, right_op, use_numexpr: bool = True) -> Any:

def where(cond, left_op, right_op, use_numexpr: bool = True) -> Any:

def set_test_mode(v: bool = True) -> None:

def _store_test_result(used_numexpr: bool) -> None:

def get_test_result() -> List[bool]:
