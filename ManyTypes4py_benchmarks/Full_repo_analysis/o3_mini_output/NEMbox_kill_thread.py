import ctypes
import inspect
import threading
import time
from typing import Type

__all__ = ['stop_thread']

def _async_raise(tid: int, exctype: Type[BaseException]) -> None:
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError('invalid thread id')
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError('PyThreadState_SetAsyncExc failed')

def stop_thread(thread: threading.Thread) -> None:
    if thread.ident is not None:
        _async_raise(thread.ident, SystemExit)
    else:
        raise ValueError("Thread does not have a valid thread identifier.")

def test() -> None:
    while True:
        print('-------')
        time.sleep(0.5)

if __name__ == '__main__':
    t = threading.Thread(target=test)
    t.start()
    time.sleep(5.2)
    print('main thread sleep finish')
    stop_thread(t)