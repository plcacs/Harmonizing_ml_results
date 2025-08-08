from typing import Dict, Any, Union

def should_log(self, logger_name: str, level: str) -> bool:
def configure_logging(log_config: Dict[str, str], disable_debug_logfile: bool, debug_log_file_path: str, colorize: bool) -> None:
def test_basic_logging(capsys: Any, module: str, level: str, logger: str, disabled_debug: bool, tmpdir: Any) -> None:
def test_debug_logfile_invalid_dir() -> None:
def test_redacted_request(capsys: Any, tmpdir: Any) -> None:
def test_that_secret_is_redacted(capsys: Any, tmpdir: Any) -> None:
