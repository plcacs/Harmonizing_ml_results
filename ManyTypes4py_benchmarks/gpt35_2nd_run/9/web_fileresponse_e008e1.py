from typing import IO, TYPE_CHECKING, Any, Awaitable, Callable, Optional, Set, Tuple

_T_OnChunkSent = Optional[Callable[[bytes], Awaitable[None]]

class FileResponse(StreamResponse):
    def __init__(self, path: PathLike, chunk_size: int = 256 * 1024, status: int = 200, reason: Optional[str] = None, headers: Optional[LooseHeaders] = None) -> None:
    def _seek_and_read(self, fobj: IO[bytes], offset: int, chunk_size: int) -> bytes:
    async def _sendfile_fallback(self, writer: AbstractStreamWriter, fobj: IO[bytes], offset: int, count: int) -> AbstractStreamWriter:
    async def _sendfile(self, request: 'BaseRequest', fobj: IO[bytes], offset: int, count: int) -> AbstractStreamWriter:
    @staticmethod
    def _etag_match(etag_value: str, etags: Set[ETag], *, weak: bool) -> bool:
    async def _not_modified(self, request: 'BaseRequest', etag_value: str, last_modified: float) -> AbstractStreamWriter:
    async def _precondition_failed(self, request: 'BaseRequest') -> AbstractStreamWriter:
    def _make_response(self, request: 'BaseRequest', accept_encoding: str) -> Tuple[_FileResponseResult, Optional[IO[bytes]], os.stat_result, Optional[str]]:
    def _get_file_path_stat_encoding(self, accept_encoding: str) -> Tuple[PathLike, os.stat_result, Optional[str]]:
    async def prepare(self, request: 'BaseRequest') -> AbstractStreamWriter:
    async def _prepare_open_file(self, request: 'BaseRequest', fobj: IO[bytes], st: os.stat_result, file_encoding: Optional[str]) -> AbstractStreamWriter:
