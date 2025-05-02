import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
import numpy as np
from nevergrad.common import typing as tp
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

class BoundChecker:
    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None) -> None:
        self.bounds: Tuple[Optional[float], Optional[float]] = (lower, upper)

    def __call__(self, value: np.ndarray) -> bool:
        for k, bound in enumerate(self.bounds):
            if bound is not None:
                if np.any(value > bound if k else value < bound):
                    return False
        return True

class FunctionInfo:
    def __init__(self, deterministic: bool = True, proxy: bool = False, metrizable: bool = True) -> None:
        self.deterministic: bool = deterministic
        self.proxy: bool = proxy
        self.metrizable: bool = metrizable

    def __repr__(self) -> str:
        diff = ','.join((f'{x}={y}' for x, y in sorted(self.__dict__.items())))
        return f'{self.__class__.__name__}({diff})'

class TemporaryDirectoryCopy(tempfile.TemporaryDirectory):
    key = 'CLEAN_COPY_DIRECTORY'

    @classmethod
    def set_clean_copy_environment_variable(cls, directory: Union[Path, str]) -> None:
        assert Path(directory).exists(), 'Directory does not exist'
        os.environ[cls.key] = str(directory)

    def __init__(self, source: Union[Path, str], dir: Optional[Union[Path, str]] = None) -> None:
        if dir is None:
            dir = os.environ.get(self.key, None)
        super().__init__(prefix='tmp_clean_copy_', dir=dir)
        self.copyname: Path = Path(self.name) / Path(source).name
        shutil.copytree(str(source), str(self.copyname))

    def __enter__(self) -> Path:
        super().__enter__()
        return self.copyname

class FailedJobError(RuntimeError):
    pass

class CommandFunction:
    def __init__(self, command: List[str], verbose: bool = False, cwd: Optional[Union[Path, str]] = None, env: Optional[Dict[str, str]] = None) -> None:
        if not isinstance(command, list):
            raise TypeError('The command must be provided as a list')
        self.command: List[str] = command
        self.verbose: bool = verbose
        self.cwd: Optional[str] = None if cwd is None else str(cwd)
        self.env: Optional[Dict[str, str]] = env

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        full_command = self.command + [str(x) for x in args] + ['--{}={}'.format(x, y) for x, y in kwargs.items()]
        if self.verbose:
            print(f'The following command is sent: {full_command}')
        outlines: List[str] = []
        with subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, cwd=self.cwd, env=self.env) as process:
            try:
                assert process.stdout is not None
                for line in iter(process.stdout.readline, b''):
                    if not line:
                        break
                    outlines.append(line.decode().strip())
                    if self.verbose:
                        print(outlines[-1], flush=True)
            except Exception:
                process.kill()
                process.wait()
                raise FailedJobError('Job got killed for an unknown reason.')
            stderr = process.communicate()[1]
            stdout = '\n'.join(outlines)
            retcode = process.poll()
            if stderr and (retcode or self.verbose):
                print(stderr.decode(), file=sys.stderr)
            if retcode:
                subprocess_error = subprocess.CalledProcessError(retcode, process.args, output=stdout, stderr=stderr)
                raise FailedJobError(stderr.decode()) from subprocess_error
        return stdout

X = tp.TypeVar('X')

class Subobjects(tp.Generic[X]):
    def __init__(self, obj: Any, base: Type[X], attribute: str) -> None:
        self.obj: Any = obj
        self.cls: Type[X] = base
        self.attribute: str = attribute

    def new(self, obj: Any) -> 'Subobjects[X]':
        return Subobjects(obj, base=self.cls, attribute=self.attribute)

    def items(self) -> Generator[Tuple[Any, X], None, None]:
        container = getattr(self.obj, self.attribute)
        if not isinstance(container, (list, dict)):
            raise TypeError('Subcaller only work on list and dict')
        iterator = enumerate(container) if isinstance(container, list) else container.items()
        for key, val in iterator:
            if isinstance(val, self.cls):
                yield (key, val)

    def _get_subobject(self, obj: Any, key: Any) -> Any:
        if isinstance(obj, self.cls):
            return getattr(obj, self.attribute)[key]
        return obj

    def apply(self, method: str, *args: Any, **kwargs: Any) -> Dict[Any, Any]:
        outputs: Dict[Any, Any] = {}
        for key, subobj in self.items():
            subargs = [self._get_subobject(arg, key) for arg in args]
            subkwargs = {k: self._get_subobject(kwarg, key) for k, kwarg in kwargs.items()}
            outputs[key] = getattr(subobj, method)(*subargs, **subkwargs)
        return outputs

def float_penalty(x: Union[bool, np.bool_, float, np.float64]) -> float:
    if isinstance(x, (bool, np.bool_)):
        return float(not x)
    elif isinstance(x, (float, np.float64)):
        return -min(0, x)
    raise TypeError(f'Only bools and floats are supported for check constaint, but got: {x} ({type(x)})')

class _ConstraintCompatibilityFunction:
    def __init__(self, func: Any) -> None:
        self.func: Any = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.func((args, kwargs))
        return out
