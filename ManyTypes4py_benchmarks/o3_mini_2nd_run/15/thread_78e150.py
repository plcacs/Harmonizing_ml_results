"""Module containing the SessionThread class."""
import threading
import uuid
import requests
import requests.exceptions as exc
from typing import Any, Dict, Tuple
from .._compat import queue
from queue import Empty  # or use queue.Empty if compatible

class SessionThread:
    def __init__(
        self,
        initialized_session: requests.sessions.Session,
        job_queue: "queue.Queue[Dict[str, Any]]",
        response_queue: "queue.Queue[Tuple[Dict[str, Any], requests.models.Response]]",
        exception_queue: "queue.Queue[Tuple[Dict[str, Any], exc.RequestException]]"
    ) -> None:
        self._session: requests.sessions.Session = initialized_session
        self._jobs: "queue.Queue[Dict[str, Any]]" = job_queue
        self._responses: "queue.Queue[Tuple[Dict[str, Any], requests.models.Response]]" = response_queue
        self._exceptions: "queue.Queue[Tuple[Dict[str, Any], exc.RequestException]]" = exception_queue
        self._create_worker()

    def _create_worker(self) -> None:
        self._worker = threading.Thread(target=self._make_request, name=str(uuid.uuid4()))
        self._worker.daemon = True
        self._worker._state = 0  # type: ignore
        self._worker.start()

    def _handle_request(self, kwargs: Dict[str, Any]) -> None:
        try:
            response: requests.models.Response = self._session.request(**kwargs)
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