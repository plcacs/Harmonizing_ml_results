from typing import Dict, Tuple, Any, cast
import numpy as np
from io import StringIO

# The fix is to explicitly annotate the dtype dict with proper types
dtype_dict: Dict[Tuple[str, str], Any] = {
    ('A', 'X'): np.int32,
    ('B', 'Y'): np.int32,
    ('B', 'Z'): np.float32
}