#!/usr/bin/env python3
"""Module implementing the Pool for :mod:`requests_toolbelt.threaded`."""

import multiprocessing
import requests
from typing import Any, Callable, Optional, Iterator, Iterable, Dict, Type
from . import thread
from .._compat import queue

class Pool(object):
    """Pool that manages the threads containing sessions.

    :param job_queue:
        The queue you're expected to use to which you should add items.
    :type job_queue: queue.Queue
    :param initializer:
        Function used to initialize an instance of ``session``.
    :type initializer: Optional[Callable[[requests.Session], requests.Session]]
    :param auth_generator:
        Function used to generate new auth credentials for the session.
    :type auth_generator: Optional[Callable[[requests.Session], requests.Session]]
    :param num_processes:
        Number of threads to create.
    :type num_processes: Optional[int]
    :param session:
    :type session: Type[requests.Session]
    """
    def __init__(
        self,
        job_queue: queue.Queue,
        initializer: Optional[Callable[[requests.Session], requests.Session]] = None,
        auth_generator: Optional[Callable[[requests.Session], requests.Session]] = None,
        num_processes: Optional[int] = None,
        session: Type[requests.Session] = requests.Session
    ) -> None:
        if num_processes is None:
            num_processes = multiprocessing.cpu_count() or 1
        if num_processes < 1:
            raise ValueError('Number of processes should at least be 1.')
        self._job_queue: queue.Queue = job_queue
        self._response_queue: queue.Queue = queue.Queue()
        self._exc_queue: queue.Queue = queue.Queue()
        self._processes: int = num_processes
        self._initializer: Callable[[requests.Session], requests.Session] = initializer or _identity
        self._auth: Callable[[requests.Session], requests.Session] = auth_generator or _identity
        self._session: Type[requests.Session] = session
        self._pool: list[thread.SessionThread] = [
            thread.SessionThread(self._new_session(), self._job_queue, self._response_queue, self._exc_queue)
            for _ in range(self._processes)
        ]

    def _new_session(self) -> requests.Session:
        return self._auth(self._initializer(self._session()))

    @classmethod
    def from_exceptions(cls, exceptions: Iterable["ThreadException"], **kwargs: Any) -> "Pool":
        """Create a :class:`~Pool` from an :class:`~ThreadException`\\ s.

        Provided an iterable that provides :class:`~ThreadException` objects,
        this classmethod will generate a new pool to retry the requests that
        caused the exceptions.

        :param exceptions:
            Iterable that returns :class:`~ThreadException`
        :type exceptions: Iterable[ThreadException]
        :param kwargs:
            Keyword arguments passed to the :class:`~Pool` initializer.
        :returns: An initialized :class:`~Pool` object.
        :rtype: Pool
        """
        job_queue: queue.Queue = queue.Queue()
        for exc in exceptions:
            job_queue.put(exc.request_kwargs)
        return cls(job_queue=job_queue, **kwargs)

    @classmethod
    def from_urls(cls, urls: Iterable[str], request_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> "Pool":
        """Create a :class:`~Pool` from an iterable of URLs.

        :param urls:
            Iterable that returns URLs with which we create a pool.
        :type urls: Iterable[str]
        :param request_kwargs:
            Dictionary of other keyword arguments to provide to the request
            method.
        :type request_kwargs: Optional[Dict[str, Any]]
        :param kwargs:
            Keyword arguments passed to the :class:`~Pool` initializer.
        :returns: An initialized :class:`~Pool` object.
        :rtype: Pool
        """
        request_dict: Dict[str, Any] = {'method': 'GET'}
        if request_kwargs:
            request_dict.update(request_kwargs)
        job_queue: queue.Queue = queue.Queue()
        for url in urls:
            job: Dict[str, Any] = request_dict.copy()
            job.update({'url': url})
            job_queue.put(job)
        return cls(job_queue=job_queue, **kwargs)

    def exceptions(self) -> Iterator["ThreadException"]:
        """Iterate over all the exceptions in the pool.

        :returns: Generator of :class:`~ThreadException`
        """
        while True:
            exc: Optional[ThreadException] = self.get_exception()
            if exc is None:
                break
            yield exc

    def get_exception(self) -> Optional["ThreadException"]:
        """Get an exception from the pool.

        :rtype: Optional[ThreadException]
        """
        try:
            request, exc = self._exc_queue.get_nowait()
        except queue.Empty:
            return None
        else:
            return ThreadException(request, exc)

    def get_response(self) -> Optional["ThreadResponse"]:
        """Get a response from the pool.

        :rtype: Optional[ThreadResponse]
        """
        try:
            request, response = self._response_queue.get_nowait()
        except queue.Empty:
            return None
        else:
            return ThreadResponse(request, response)

    def responses(self) -> Iterator["ThreadResponse"]:
        """Iterate over all the responses in the pool.

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

class ThreadProxy(object):
    proxied_attr: Optional[str] = None

    def __getattr__(self, attr: str) -> Any:
        """Proxy attribute accesses to the proxied object."""
        get = object.__getattribute__
        if attr not in self.attrs:  # type: ignore[attr-defined]
            response = get(self, self.proxied_attr)  # type: ignore[arg-type]
            return getattr(response, attr)
        else:
            return get(self, attr)

class ThreadResponse(ThreadProxy):
    """A wrapper around a requests Response object.

    This will proxy most attribute access actions to the Response object. For
    example, if you wanted the parsed JSON from the response, you might do:

    .. code-block:: python

        thread_response = pool.get_response()
        json = thread_response.json()

    """
    proxied_attr: str = 'response'
    attrs = frozenset(['request_kwargs', 'response'])

    def __init__(self, request_kwargs: Dict[str, Any], response: requests.Response) -> None:
        self.request_kwargs: Dict[str, Any] = request_kwargs
        self.response: requests.Response = response

class ThreadException(ThreadProxy):
    """A wrapper around an exception raised during a request.

    This will proxy most attribute access actions to the exception object. For
    example, if you wanted the message from the exception, you might do:

    .. code-block:: python

        thread_exc = pool.get_exception()
        msg = thread_exc.message

    """
    proxied_attr: str = 'exception'
    attrs = frozenset(['request_kwargs', 'exception'])

    def __init__(self, request_kwargs: Dict[str, Any], exception: Exception) -> None:
        self.request_kwargs: Dict[str, Any] = request_kwargs
        self.exception: Exception = exception

def _identity(session_obj: Any) -> Any:
    return session_obj

__all__ = ['ThreadException', 'ThreadResponse', 'Pool']