from typing import Dict, Any, Union, List, Tuple, Optional

def test_log_filter() -> None:
    rules: Dict[str, str]
    filter_: LogFilter
    assert filter_.should_log('test', 'DEBUG') is False

def test_basic_logging(capsys: Any, module: str, level: str, logger: str, disabled_debug: bool, tmpdir: Any) -> None:
    log: Any
    captured: Any
    no_log: bool

def test_debug_logfile_invalid_dir() -> None:
    pass

def test_redacted_request(capsys: Any, tmpdir: Any) -> None:
    token: str
    log: Any
    captured: Any

def test_that_secret_is_redacted(capsys: Any, tmpdir: Any) -> None:
    secret: str
    data: str
    log: Any
    captured: Any
