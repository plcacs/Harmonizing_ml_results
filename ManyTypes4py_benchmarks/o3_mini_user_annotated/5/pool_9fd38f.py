#!/usr/bin/env python
"""
Module implementing the Pool for :mod:`requests_toolbelt.threaded`.
"""

import multiprocessing
import requests
from typing import Any, Callable, Dict, Iterator, Iterable, Optional, Type, TypeVar, List

from . import thread
from .._compat import queue

T = TypeVar("T")


def _identity(session_obj: T) -> T:
    return session_obj


class Pool:
    def __init__(
        self,
        job_queue: queue.Queue,
        initializer: Optional[Callable[[Any], Any]] = None,
        auth_generator: Optional[Callable[[Any], Any]] = None,
        num_processes: Optional[int] = None,
        session: Type[requests.Session] = requests.Session,
    ) -> None:
        if num_processes is None:
            num_processes = multiprocessing.cpu_count() or 1

        if num_processes < 1:
            raise ValueError("Number of processes should at least be 1.")

        self._job_queue: queue.Queue = job_queue
        self._response_queue: queue.Queue = queue.Queue()
        self._exc_queue: queue.Queue = queue.Queue()
        self._processes: int = num_processes
        self._initializer: Callable[[Any], Any] = initializer or _identity
        self._auth: Callable[[Any], Any] = auth_generator or _identity
        self._session: Type[requests.Session] = session
        self._pool: List[thread.SessionThread] = [
            thread.SessionThread(
                self._new_session(),
                self._job_queue,
                self._response_queue,
                self._exc_queue,
            )
            for _ in range(self._processes)
        ]

    def _new_session(self) -> requests.Session:
        return self._auth(self._initializer(self._session()))

    @classmethod
    def from_exceptions(
        cls,
        exceptions: Iterable["ThreadException"],
        **kwargs: Any,
    ) -> "Pool":
        r"""
        Create a :class:`~Pool` from :class:`~ThreadException`\ s.

        Provided an iterable that provides :class:`~ThreadException` objects,
        this classmethod will generate a new pool to retry the requests that
        caused the exceptions.
        """
        job_queue: queue.Queue = queue.Queue()
        for exc in exceptions:
            job_queue.put(exc.request_kwargs)

        return cls(job_queue=job_queue, **kwargs)

    @classmethod
    def from_urls(
        cls,
        urls: Iterable[str],
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "Pool":
        """
        Create a :class:`~Pool` from an iterable of URLs.

        :param urls: Iterable that returns URLs with which we create a pool.
        :param request_kwargs: Dictionary of other keyword arguments to provide to the request method.
        :param kwargs: Keyword arguments passed to the :class:`~Pool` initializer.
        :returns: An initialized :class:`~Pool` object.
        """
        request_dict: Dict[str, Any] = {"method": "GET"}
        request_dict.update(request_kwargs or {})
        job_queue: queue.Queue = queue.Queue()
        for url in urls:
            job = request_dict.copy()
            job.update({"url": url})
            job_queue.put(job)

        return cls(job_queue=job_queue, **kwargs)

    def exceptions(self) -> Iterator["ThreadException"]:
        """
        Iterate over all the exceptions in the pool.
        :returns: Generator of :class:`~ThreadException`
        """
        while True:
            exc: Optional[ThreadException] = self.get_exception()
            if exc is None:
                break
            yield exc

    def get_exception(self) -> Optional["ThreadException"]:
        """
        Get an exception from the pool.
        :rtype: :class:`~ThreadException`
        """
        try:
            (request, exc) = self._exc_queue.get_nowait()
        except queue.Empty:
            return None
        else:
            return ThreadException(request, exc)

    def get_response(self) -> Optional["ThreadResponse"]:
        """
        Get a response from the pool.
        :rtype: :class:`~ThreadResponse`
        """
        try:
            (request, response) = self._response_queue.get_nowait()
        except queue.Empty:
            return None
        else:
            return ThreadResponse(request, response)

    def responses(self) -> Iterator["ThreadResponse"]:
        """
        Iterate over all the responses in the pool.
        :returns: Generator of :class:`~ThreadResponse`
        """
        while True:
            resp: Optional[ThreadResponse] = self.get_response()
            if resp is None:
                break
            yield resp

    def join_all(self) -> None:
        """Join all the threads to the master thread."""
        for session_thread in self._pool:
            session_thread.join()


class ThreadProxy:
    proxied_attr: str = ""
    attrs: frozenset = frozenset()

    def __getattr__(self, attr: str) -> Any:
        """
        Proxy attribute accesses to the proxied object.
        """
        get = object.__getattribute__
        if attr not in self.attrs:
            response = get(self, self.proxied_attr)
            return getattr(response, attr)
        else:
            return get(self, attr)


class ThreadResponse(ThreadProxy):
    """
    A wrapper around a requests Response object.
    This will proxy most attribute access actions to the Response object.
    """
    proxied_attr: str = "response"
    attrs: frozenset = frozenset(["request_kwargs", "response"])

    def __init__(self, request_kwargs: Dict[str, Any], response: requests.Response) -> None:
        #: The original keyword arguments provided to the queue
        self.request_kwargs: Dict[str, Any] = request_kwargs
        #: The wrapped response
        self.response: requests.Response = response


class ThreadException(ThreadProxy):
    """
    A wrapper around an exception raised during a request.
    This will proxy most attribute access actions to the exception object.
    """
    proxied_attr: str = "exception"
    attrs: frozenset = frozenset(["request_kwargs", "exception"])

    def __init__(self, request_kwargs: Dict[str, Any], exception: Exception) -> None:
        #: The original keyword arguments provided to the queue
        self.request_kwargs: Dict[str, Any] = request_kwargs
        #: The captured and wrapped exception
        self.exception: Exception = exception


__all__ = ["ThreadException", "ThreadResponse", "Pool"]