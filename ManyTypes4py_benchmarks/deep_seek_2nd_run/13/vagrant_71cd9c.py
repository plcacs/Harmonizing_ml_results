import os
import threading
import subprocess
from typing import Optional, Callable, Tuple, Any, Union, List
from .logger import Log
from .contexts import vagrant_root
from .helpers import create_subprocess
PIPE = subprocess.PIPE

class VagrantBase(threading.Thread):
    """Spawn vagrant shell to execute commands (base class)
    """

    def __init__(self, callback: Callable[[Tuple[bool, bytes, bytes]], None], vagrant_root: str, machine: Optional[str] = None) -> None:
        super(VagrantBase, self).__init__()
        self.machine: str = machine if machine is not None else 'default'
        self.callback: Callable[[Tuple[bool, bytes, bytes]], None] = callback
        self.vagrant_root: str = vagrant_root

    def wait_answer(self, args: List[str]) -> None:
        """Wait for an answer from the subprocess
        """
        with vagrant_root(self.vagrant_root):
            proc: subprocess.Popen = create_subprocess(args, stdout=PIPE, stderr=PIPE, cwd=os.getcwd())
            output: bytes
            error: bytes
            output, error = proc.communicate()
        self.callback((proc.poll() == 0, output, error))

class VagrantInit(VagrantBase):
    """Init a new vagrant environment with the given box
    """

    def __init__(self, callback: Callable[[Tuple[bool, bytes, bytes]], None], vagrant_root: str, box: str) -> None:
        super(VagrantInit, self).__init__(callback, vagrant_root)
        self.box: str = box
        self.start()

    def run(self) -> None:
        """Init the vagrant machine
        """
        self.wait_answer(['vagrant', 'init', self.box])

class VagrantUp(VagrantBase):
    """Start a vagrant box
    """

    def __init__(self, callback: Callable[[Tuple[bool, bytes, bytes]], None], vagrant_root: str, machine: Optional[str] = None) -> None:
        super(VagrantUp, self).__init__(callback, vagrant_root, machine)
        self.start()

    def run(self) -> None:
        """Start the vagrant box machine
        """
        self.wait_answer(['vagrant', 'up', self.machine])

class VagrantReload(VagrantBase):
    """Reload a vagrant box
    """

    def __init__(self, callback: Callable[[Tuple[bool, bytes, bytes]], None], vagrant_root: str, machine: Optional[str] = None) -> None:
        super(VagrantReload, self).__init__(callback, vagrant_root, machine)
        self.start()

    def run(self) -> None:
        """Reload the vagrant box machine
        """
        self.wait_answer(['vagrant', 'reload', self.machine])

class VagrantStatus(VagrantBase):
    """Check vagrant box status
    """

    def __init__(self, callback: Callable[[Tuple[bool, Union[bool, bytes]]], None], vagrant_root: str, machine: Optional[str] = None, full: bool = False) -> None:
        super(VagrantStatus, self).__init__(callback, vagrant_root, machine)
        self.full: bool = full
        self.start()

    def run(self) -> None:
        """Check the vagrant box machine status
        """
        args: List[str] = ['vagrant', 'status', self.machine]
        with vagrant_root(self.vagrant_root):
            proc: subprocess.Popen = create_subprocess(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
        output: bytes
        error: bytes
        output, error = proc.communicate()
        if proc.poll() != 0:
            self.callback((False, error))
        else:
            running: bool = b'running' in output
            self.callback((True, running if not self.full else output))

class VagrantSSH(VagrantBase):
    """Execute a remote SSH command in a vagrant box
    """

    def __init__(self, callback: Callable[[Tuple[bool, bytes, bytes]], None], vagrant_root: str, cmd: str, machine: Optional[str] = None) -> None:
        super(VagrantSSH, self).__init__(callback, vagrant_root, machine)
        self.cmd: str = cmd
        self.start()

    def run(self) -> None:
        """Execute a command through SSH in a vagrant box machine
        """
        self.wait_answer(['vagrant', 'ssh', self.machine, '-c', self.cmd])

class VagrantIPAddress(object):
    """Get back the remote guest IP address in synchronous way
    """

    def __init__(self, root: str, machine: Optional[str] = None, iface: str = 'eth1') -> None:
        with vagrant_root(root):
            cmd: str = 'python -c "import socket, fcntl, struct;s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);print(socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack(b\'256s\', b\'{}\'))[20:24]))"\n'.format(iface)
            proc: subprocess.Popen = create_subprocess(['vagrant', 'ssh', machine, '-c', cmd], stdout=PIPE, stderr=PIPE, cwd=os.getcwd())
        output: bytes
        error: bytes
        output, error = proc.communicate()
        if proc.poll() != 0:
            self.ip_address: Optional[bytes] = None
        else:
            self.ip_address: Optional[bytes] = output

class VagrantIPAddressGlobal(object):
    """Get back the remote guest IP address in synchronous way from global
    """

    def __init__(self, machine: str, iface: str = 'eth1') -> None:
        cmd: str = 'python -c "import socket, fcntl, struct;s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);print(socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack(b\'256s\', b\'{}\'))[20:24]))"\n'.format(iface)
        proc: subprocess.Popen = create_subprocess(['vagrant', 'ssh', machine, '-c', cmd], stdout=PIPE, stderr=PIPE, cwd=os.getcwd())
        output: bytes
        error: bytes
        output, error = proc.communicate()
        if proc.poll() != 0:
            self.ip_address: Optional[bytes] = None
        else:
            self.ip_address: Optional[bytes] = output

class VagrantMachineGlobalInfo(object):
    """Get vagrant machine information looking up in global stats
    """

    def __init__(self, machine: str) -> None:
        self.machine: str = machine
        self.status: str = ''
        self.machine_id: str = ''
        args: Tuple[str, str] = ('vagrant', 'global-status')
        p: subprocess.Popen = create_subprocess(args, stdout=PIPE, stderr=PIPE)
        if p is None:
            raise RuntimeError('vagrant command not found')
        output: bytes
        err: bytes
        output, err = p.communicate()
        if err:
            raise RuntimeError(err)
        for line in output.splitlines()[2:]:
            if not line:
                continue
            if line.startswith(b'The'):
                break
            data: List[bytes] = line.split()
            if not data:
                continue
            if data[1].decode('utf8') == machine:
                self.machine_id: str = data[0].decode('utf8')
                self.status: str = data[3].decode('utf8')
                self.directory: str = data[4].decode('utf8')
                break

class VagrantStartMachine(object):
    """Start a vagrant machine using it's global ID
    """

    def __init__(self, machine: str, directory: str) -> None:
        with vagrant_root(directory):
            args: Tuple[str, str, str] = ('vagrant', 'up', machine)
            p: subprocess.Popen = create_subprocess(args, stdout=PIPE, stderr=PIPE, cwd=os.getcwd())
            if p is None:
                raise RuntimeError('vagrant command not found')
            output: bytes
            err: bytes
            output, err = p.communicate()
            Log.info(output.decode('utf8'))
            if err:
                info: VagrantMachineGlobalInfo = VagrantMachineGlobalInfo(machine)
                if info.status != 'running':
                    raise RuntimeError(err)
                Log.error(err.decode('utf8'))
