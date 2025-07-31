import multiprocessing
import requests
from typing import Any, Callable, Optional, Generator, List, Iterable
from . import thread
from .._compat import queue


class Pool(object):
    def __init__(
        self,
        job_queue: queue.Queue,
        initializer: Optional[Callable[[Any], Any]] = None,
        auth_generator: Optional[Callable[[Any], Any]] = None,
        num_processes: Optional[int] = None,
        session: Callable[[], requests.Session] = requests.Session
    ) -> None:
        if num_processes is None:
            num_processes = multiprocessing.cpu_count() or 1
        if num_processes < 1:
            raise ValueError('Number of processes should at least be 1.')
        self._job_queue: queue.Queue = job_queue
        self._response_queue: queue.Queue = queue.Queue()
        self._exc_queue: queue.Queue = queue.Queue()
        self._processes: int = num_processes
        self._initializer: Callable[[Any], Any] = initializer or _identity
        self._auth: Callable[[Any], Any] = auth_generator or _identity
        self._session: Callable[[], requests.Session] = session
        self._pool: List[thread.SessionThread] = [
            thread.SessionThread(self._new_session(), self._job_queue, self._response_queue, self._exc_queue)
            for _ in range(self._processes)
        ]

    def _new_session(self) -> Any:
        return self._auth(self._initializer(self._session()))

    @classmethod
    def from_exceptions(cls, exceptions: Iterable[Any], **kwargs: Any) -> "Pool":
        job_queue: queue.Queue = queue.Queue()
        for exc in exceptions:
            job_queue.put(exc.request_kwargs)
        return cls(job_queue=job_queue, **kwargs)

    @classmethod
    def from_urls(cls, urls: Iterable[str], request_kwargs: Optional[dict] = None, **kwargs: Any) -> "Pool":
        request_dict: dict = {'method': 'GET'}
        request_dict.update(request_kwargs or {})
        job_queue: queue.Queue = queue.Queue()
        for url in urls:
            job = request_dict.copy()
            job.update({'url': url})
            job_queue.put(job)
        return cls(job_queue=job_queue, **kwargs)

    def exceptions(self) -> Generator["ThreadException", None, None]:
        while True:
            exc = self.get_exception()
            if exc is None:
                break
            yield exc

    def get_exception(self) -> Optional["ThreadException"]:
        try:
            request, exc = self._exc_queue.get_nowait()
        except queue.Empty:
            return None
        else:
            return ThreadException(request, exc)

    def get_response(self) -> Optional["ThreadResponse"]:
        try:
            request, response = self._response_queue.get_nowait()
        except queue.Empty:
            return None
        else:
            return ThreadResponse(request, response)

    def responses(self) -> Generator["ThreadResponse", None, None]:
        while True:
            resp = self.get_response()
            if resp is None:
                break
            yield resp

    def join_all(self) -> None:
        for session_thread in self._pool:
            session_thread.join()


class ThreadProxy(object):
    proxied_attr: Optional[str] = None

    def __getattr__(self, attr: str) -> Any:
        get = object.__getattribute__
        if attr not in self.attrs:
            response = get(self, self.proxied_attr)
            return getattr(response, attr)
        else:
            return get(self, attr)


class ThreadResponse(ThreadProxy):
    proxied_attr: str = 'response'
    attrs = frozenset(['request_kwargs', 'response'])

    def __init__(self, request_kwargs: Any, response: requests.Response) -> None:
        self.request_kwargs: Any = request_kwargs
        self.response: requests.Response = response


class ThreadException(ThreadProxy):
    proxied_attr: str = 'exception'
    attrs = frozenset(['request_kwargs', 'exception'])

    def __init__(self, request_kwargs: Any, exception: Exception) -> None:
        self.request_kwargs: Any = request_kwargs
        self.exception: Exception = exception


def _identity(session_obj: Any) -> Any:
    return session_obj


__all__ = ['ThreadException', 'ThreadResponse', 'Pool']