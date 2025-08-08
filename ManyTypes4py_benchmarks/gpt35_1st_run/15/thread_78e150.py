import threading
import uuid
from queue import Queue
from typing import Any, Tuple
from requests import Session
from requests.exceptions import RequestException

class SessionThread:
    def __init__(self, initialized_session: Session, job_queue: Queue, response_queue: Queue, exception_queue: Queue) -> None:
        self._session: Session = initialized_session
        self._jobs: Queue = job_queue
        self._responses: Queue = response_queue
        self._exceptions: Queue = exception_queue
        self._create_worker()

    def _create_worker(self) -> None:
        self._worker: threading.Thread = threading.Thread(target=self._make_request, name=str(uuid.uuid4()))
        self._worker.daemon = True
        self._worker._state = 0
        self._worker.start()

    def _handle_request(self, kwargs: dict) -> None:
        try:
            response = self._session.request(**kwargs)
        except RequestException as e:
            self._exceptions.put((kwargs, e))
        else:
            self._responses.put((kwargs, response))
        finally:
            self._jobs.task_done()

    def _make_request(self) -> None:
        while True:
            try:
                kwargs = self._jobs.get_nowait()
            except queue.Empty:
                break
            self._handle_request(kwargs)

    def is_alive(self) -> bool:
        return self._worker.is_alive()

    def join(self) -> None:
        self._worker.join()
