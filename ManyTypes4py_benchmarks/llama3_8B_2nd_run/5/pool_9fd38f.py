"""Module implementing the Pool for :mod:``requests_toolbelt.threaded``."""
import multiprocessing
import requests
from . import thread
from .._compat import queue

class Pool(object):
    """Pool that manages the threads containing sessions.

    :param queue:
        The queue you're expected to use to which you should add items.
    :type queue: queue.Queue
    :param initializer:
        Function used to initialize an instance of ``session``.
    :type initializer: callable
    :param auth_generator:
        Function used to generate new auth credentials for the session.
    :type auth_generator: callable
    :param int num_processes:
        Number of threads to create.
    :param session:
    :type session: requests.Session
    """

    def __init__(self, job_queue: queue.Queue, initializer: callable = None, auth_generator: callable = None, num_processes: int = None, session: requests.Session = requests.Session) -> None:
        if num_processes is None:
            num_processes = multiprocessing.cpu_count() or 1
        if num_processes < 1:
            raise ValueError('Number of processes should at least be 1.')
        self._job_queue = job_queue
        self._response_queue = queue.Queue()
        self._exc_queue = queue.Queue()
        self._processes = num_processes
        self._initializer = initializer or _identity
        self._auth = auth_generator or _identity
        self._session = session
        self._pool = [thread.SessionThread(self._new_session(), self._job_queue, self._response_queue, self._exc_queue) for _ in range(self._processes)]

    # ... rest of the code ...

class ThreadProxy(object):
    proxied_attr: str
    attrs: frozenset

    def __getattr__(self, attr: str) -> object:
        """Proxy attribute accesses to the proxied object."""
        get = object.__getattribute__
        if attr not in self.attrs:
            response = get(self, self.proxied_attr)
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
    attrs: frozenset = frozenset(['request_kwargs', 'response'])

    def __init__(self, request_kwargs: dict, response: requests.Response) -> None:
        self.request_kwargs = request_kwargs
        self.response = response

class ThreadException(ThreadProxy):
    """A wrapper around an exception raised during a request.

    This will proxy most attribute access actions to the exception object. For
    example, if you wanted the message from the exception, you might do:

    .. code-block:: python

        thread_exc = pool.get_exception()
        msg = thread_exc.message

    """
    proxied_attr: str = 'exception'
    attrs: frozenset = frozenset(['request_kwargs', 'exception'])

    def __init__(self, request_kwargs: dict, exception: Exception) -> None:
        self.request_kwargs = request_kwargs
        self.exception = exception

def _identity(session_obj: requests.Session) -> requests.Session:
    return session_obj
__all__ = ['ThreadException', 'ThreadResponse', 'Pool']