from __future__ import annotations
import logging.handlers
import sys
import traceback
from types import TracebackType
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union

import orjson
from prefect.serializers import JSONSerializer

ExceptionInfoType = Union[
    Tuple[Type[BaseException], BaseException, Optional[TracebackType]],
    Tuple[None, None, None],
]


def format_exception_info(
    exc_info: ExceptionInfoType,
) -> Dict[str, Optional[str]]:
    if exc_info[0] is None:
        return {}
    exception_type, exception_obj, exception_traceback = exc_info
    return {
        'type': exception_type.__name__,
        'message': str(exception_obj),
        'traceback': (
            ''.join(traceback.format_tb(exception_traceback))
            if exception_traceback
            else None
        ),
    }


class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON string.

    The format may be specified as "pretty" to format the JSON with indents and
    newlines.
    """

    def __init__(
        self,
        fmt: Literal["pretty", "default"],
        dmft: Any,
        style: str,
    ) -> None:
        super().__init__()
        if fmt not in ['pretty', 'default']:
            raise ValueError("Format must be either 'pretty' or 'default'.")
        dumps_kwargs: Dict[str, Any] = {'option': orjson.OPT_INDENT_2} if fmt == 'pretty' else {}
        self.serializer = JSONSerializer(jsonlib='orjson', dumps_kwargs=dumps_kwargs)

    def format(self, record: logging.LogRecord) -> str:
        record_dict = record.__dict__.copy()
        record_dict.setdefault('severity', record.levelname)
        if record.exc_info:
            record_dict['exc_info'] = format_exception_info(record.exc_info)
        log_json_bytes: bytes = self.serializer.dumps(record_dict)
        return log_json_bytes.decode()


class PrefectFormatter(logging.Formatter):

    def __init__(
        self,
        format: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = '%',
        validate: bool = True,
        *,
        defaults: Optional[Dict[str, Any]] = None,
        task_run_fmt: Optional[str] = None,
        flow_run_fmt: Optional[str] = None,
    ) -> None:
        """
        Implementation of the standard Python formatter with support for multiple
        message formats.
        """
        init_kwargs: Dict[str, Any] = {}
        style_kwargs: Dict[str, Any] = {}
        if sys.version_info >= (3, 10):
            init_kwargs['defaults'] = defaults
            style_kwargs['defaults'] = defaults
        init_kwargs['validate'] = validate
        super().__init__(format, datefmt, style, **init_kwargs)
        self.flow_run_fmt: Optional[str] = flow_run_fmt
        self.task_run_fmt: Optional[str] = task_run_fmt
        style_class = type(self._style)
        self._flow_run_style = style_class(flow_run_fmt, **style_kwargs) if flow_run_fmt else self._style
        self._task_run_style = style_class(task_run_fmt, **style_kwargs) if task_run_fmt else self._style
        if validate:
            self._flow_run_style.validate()
            self._task_run_style.validate()

    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.name == 'prefect.flow_runs':
            style = self._flow_run_style
        elif record.name == 'prefect.task_runs':
            style = self._task_run_style
        else:
            style = self._style
        return style.format(record)