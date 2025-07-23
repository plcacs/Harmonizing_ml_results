"""Module containing the SessionThread class."""
import threading
import uuid
import requests.exceptions as exc
from .._compat import queue
from queue import Queue
from requests import Session
from requests.models import Response
from typing import Any, Dict, Tuple

class SessionThread(object):

    def __init__(self, initialized_session: Session, job_queue: Queue, response_queue: Queue, exception_queue: Queue) -> None:
        self._session: Session = initialized_session
        self._jobs: Queue = job_queue
        self._create_worker()
        self._responses: Queue = response_queue
        self._exceptions: Queue = exception_queue

    def _create_worker(self) -> None:
        self._worker: threading.Thread = threading.Thread(target=self._make_request, name=str(uuid.uuid4()))
        self._worker.daemon = True
        self._worker._state = 0
        self._worker.start()

    def _handle_request(self, kwargs: Dict[str, Any]) -> None:
        try:
            response: Response = self._session.request(**kwargs)
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
