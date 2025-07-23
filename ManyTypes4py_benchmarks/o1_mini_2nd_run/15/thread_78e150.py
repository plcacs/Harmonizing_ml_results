"""Module containing the SessionThread class."""
import threading
import uuid
import requests.exceptions as exc
from .._compat import queue
from typing import Any, Dict, Tuple
import requests

class SessionThread(object):

    def __init__(
        self,
        initialized_session: requests.Session,
        job_queue: queue.Queue[Dict[str, Any]],
        response_queue: queue.Queue[Tuple[Dict[str, Any], requests.Response]],
        exception_queue: queue.Queue[Tuple[Dict[str, Any], exc.RequestException]]
    ) -> None:
        self._session = initialized_session
        self._jobs = job_queue
        self._create_worker()
        self._responses = response_queue
        self._exceptions = exception_queue

    def _create_worker(self) -> None:
        self._worker: threading.Thread = threading.Thread(
            target=self._make_request, name=str(uuid.uuid4())
        )
        self._worker.daemon = True
        self._worker._state = 0
        self._worker.start()

    def _handle_request(self, kwargs: Dict[str, Any]) -> None:
        try:
            response: requests.Response = self._session.request(**kwargs)
        except exc.RequestException as e:
            self._exceptions.put((kwargs, e))
        else:
            self._responses.put((kwargs, response))
        finally:
            self._jobs.task_done()

    def _make_request(self) -> None:
        while True:
            try:
                kwargs: Dict[str, Any] = self._jobs.get_nowait()
            except queue.Empty:
                break
            self._handle_request(kwargs)

    def is_alive(self) -> bool:
        """Proxy to the thread's ``is_alive`` method."""
        return self._worker.is_alive()

    def join(self) -> None:
        """Join this thread to the master thread."""
        self._worker.join()
