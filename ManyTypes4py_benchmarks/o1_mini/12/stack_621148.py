""" Stack and trace extraction utilities.

This code is heavilly based on the raven-python from the Sentry Team.

:copyright: (c) 2010-2012 by the Sentry Team, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""
import linecache
import sys
from typing import Any, Dict, Optional, Tuple, List, Union
from types import FrameType


def _getitem_from_frame(f_locals: Any, key: str, default: Any = None) -> Any:
    """
    f_locals is not guaranteed to have .get(), but it will always
    support __getitem__. Even if it doesn't, we return ``default``.
    """
    try:
        return f_locals[key]
    except Exception:
        return default


def to_dict(dictish: Any) -> Dict[Any, Any]:
    """
    Given something that closely resembles a dictionary, we attempt
    to coerce it into a propery dictionary.
    """
    if hasattr(dictish, 'keys'):
        method = dictish.keys
    else:
        raise ValueError(dictish)
    return {k: dictish[k] for k in method()}


def get_lines_from_file(
    filename: str,
    lineno: int,
    context_lines: int,
    loader: Optional[Any] = None,
    module_name: Optional[str] = None
) -> Tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
    """
    Returns context_lines before and after lineno from file.
    Returns (pre_context_lineno, pre_context, context_line, post_context).
    """
    source: Optional[List[str]] = None
    if loader is not None and hasattr(loader, 'get_source'):
        try:
            source_str = loader.get_source(module_name)  # type: ignore
            source = source_str.splitlines()
        except ImportError:
            source = None
    if source is None:
        try:
            source = linecache.getlines(filename)
        except OSError:
            return (None, None, None)
    if not source:
        return (None, None, None)
    lower_bound = max(0, lineno - context_lines)
    upper_bound = min(lineno + 1 + context_lines, len(source))
    try:
        pre_context = [line.strip('\r\n') for line in source[lower_bound:lineno]]
        context_line = source[lineno].strip('\r\n')
        post_context = [line.strip('\r\n') for line in source[lineno + 1:upper_bound]]
    except IndexError:
        return (None, None, None)
    return (pre_context, context_line, post_context)


def get_frame_locals(frame: FrameType) -> Optional[Dict[str, Any]]:
    f_locals = getattr(frame, 'f_locals', None)
    if not f_locals:
        return None
    if not isinstance(f_locals, dict):
        try:
            f_locals = to_dict(f_locals)
        except Exception:
            return None
    f_vars: Dict[str, Any] = {}
    f_size = 0
    for key, value in f_locals.items():
        v_size = len(repr(value))
        if v_size + f_size < 4096:
            f_vars[key] = value
            f_size += v_size
    return f_vars


def get_stack_info(frame: Union[FrameType, Tuple[FrameType, int]]) -> Dict[str, Any]:
    frame_result = get_trace_info(frame)
    abs_path: Optional[str] = frame_result.get('abs_path')
    lineno: Optional[int] = frame_result['lineno']
    module_name: Optional[str] = frame_result['module']
    f_globals = getattr(frame, 'f_globals', {}) if isinstance(frame, FrameType) else {}
    loader = _getitem_from_frame(f_globals, '__loader__')
    if lineno is not None and abs_path:
        line_data = get_lines_from_file(abs_path, lineno - 1, 5, loader, module_name)
        pre_context, context_line, post_context = line_data
        frame_result.update({'pre_context': pre_context, 'context_line': context_line, 'post_context': post_context})
    f_vars = get_frame_locals(frame) if isinstance(frame, FrameType) else None
    if f_vars:
        frame_result['vars'] = f_vars
    return frame_result


def get_trace_info(frame: Union[FrameType, Tuple[FrameType, int]]) -> Dict[str, Any]:
    if isinstance(frame, (list, tuple)):
        frame, lineno = frame
    else:
        lineno = frame.f_lineno
    f_globals = getattr(frame, 'f_globals', {}) if isinstance(frame, FrameType) else {}
    f_code = getattr(frame, 'f_code', None) if isinstance(frame, FrameType) else None
    module_name = _getitem_from_frame(f_globals, '__name__') if isinstance(frame, FrameType) else None
    if f_code:
        abs_path = f_code.co_filename
        function = f_code.co_name
    else:
        abs_path = None
        function = None
    try:
        base_module = module_name.split('.', 1)[0] if module_name else None
        base_filename = sys.modules[base_module].__file__ if base_module and base_module in sys.modules else None
        assert base_filename, 'Could not build basename from module name'
        filename = abs_path.split(base_filename.rsplit('/', 2)[0], 1)[-1].lstrip('/') if base_filename and abs_path else abs_path
    except Exception:
        filename = abs_path
    if not filename:
        filename = abs_path
    return {
        'runtime_id': id(f_code) if f_code else None,
        'abs_path': abs_path,
        'filename': filename,
        'module': module_name or None,
        'function': function or '<unknown>',
        'lineno': lineno
    }


def get_stack_from_frame(frame: FrameType) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    while frame:
        frame_result = get_stack_info(frame)
        result.append(frame_result)
        frame = frame.f_back
    return result[::-1]


def get_trace_from_frame(frame: FrameType) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    while frame:
        stack_result = get_trace_info(frame)
        result.append(stack_result)
        frame = frame.f_back
    return result[::-1]
