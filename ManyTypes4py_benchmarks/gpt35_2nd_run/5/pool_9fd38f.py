from typing import Callable, Iterable, Dict

class Pool:
    def __init__(self, job_queue: queue.Queue, initializer: Callable = None, auth_generator: Callable = None, num_processes: int = None, session: requests.Session = requests.Session):
    
    @classmethod
    def from_exceptions(cls, exceptions: Iterable, **kwargs) -> 'Pool':
    
    @classmethod
    def from_urls(cls, urls: Iterable, request_kwargs: Dict = None, **kwargs) -> 'Pool':
    
    def exceptions(self) -> Iterable[ThreadException]:
    
    def get_exception(self) -> ThreadException:
    
    def get_response(self) -> ThreadResponse:
    
    def responses(self) -> Iterable[ThreadResponse]:
    
    def join_all(self):
    
class ThreadProxy:
    def __getattr__(self, attr: str):
    
class ThreadResponse(ThreadProxy):
    def __init__(self, request_kwargs: Dict, response):
    
class ThreadException(ThreadProxy):
    def __init__(self, request_kwargs: Dict, exception):
    
def _identity(session_obj) -> requests.Session:
