from typing import Iterator, Any, Tuple, List, Dict, Union

def cd(path: str) -> Iterator[None]:
def basic_app(tmpdir: Any) -> str:
def config() -> Config:
def unused_tcp_port() -> int:
def http_session() -> HTTPFetcher:
def local_server_factory(unused_tcp_port: int) -> Iterator[Tuple[ThreadedLocalServer, int]]:
def sample_app() -> app.Chalice:
def test_has_thread_safe_current_request(config: Config, sample_app: app.Chalice, local_server_factory: Iterator[Tuple[ThreadedLocalServer, int]]) -> None:
def test_can_accept_get_request(config: Config, sample_app: app.Chalice, local_server_factory: Iterator[Tuple[ThreadedLocalServer, int]]) -> None:
def test_can_get_unicode_string_content_length(config: Config, local_server_factory: Iterator[Tuple[ThreadedLocalServer, int]]) -> None:
def test_can_accept_options_request(config: Config, sample_app: app.Chalice, local_server_factory: Iterator[Tuple[ThreadedLocalServer, int]]) -> None:
def test_can_accept_multiple_options_request(config: Config, sample_app: app.Chalice, local_server_factory: Iterator[Tuple[ThreadedLocalServer, int]]) -> None:
def test_can_accept_multiple_connections(config: Config, sample_app: app.Chalice, local_server_factory: Iterator[Tuple[ThreadedLocalServer, int]]) -> None:
def test_can_import_env_vars(unused_tcp_port: int, http_session: HTTPFetcher) -> None:
def _wait_for_server_ready(process: Any) -> None:
def _assert_env_var_loaded(port_number: int, http_session: HTTPFetcher) -> None:
def test_can_reload_server(unused_tcp_port: int, basic_app: str, http_session: HTTPFetcher) -> None:
