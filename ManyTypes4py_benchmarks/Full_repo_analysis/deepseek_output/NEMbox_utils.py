import os
import platform
import subprocess
import hashlib
from collections import OrderedDict
from typing import List, Union, Dict, Any, Optional

__all__ = ['utf8_data_to_file', 'notify', 'uniq', 'create_dir', 'create_file', 'md5']

def md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def mkdir(path: str) -> bool:
    try:
        os.mkdir(path)
        return True
    except OSError:
        return False

def create_dir(path: str) -> bool:
    if not os.path.exists(path):
        return mkdir(path)
    elif os.path.isdir(path):
        return True
    else:
        os.remove(path)
        return mkdir(path)

def create_file(path: str, default: str = '\n') -> None:
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(default)

def uniq(arr: List[Any]) -> List[Any]:
    return list(OrderedDict.fromkeys(arr).keys())

def utf8_data_to_file(f: Any, data: Union[str, bytes]) -> None:
    if hasattr(data, 'decode'):
        f.write(data.decode('utf-8'))
    else:
        f.write(data)

def notify_command_osx(msg: str, msg_type: int, duration_time: Optional[int] = None) -> List[bytes]:
    command = ['/usr/bin/osascript', '-e']
    tpl = 'display notification "{}" {} with title "musicbox"'
    sound = 'sound name "/System/Library/Sounds/Ping.aiff"' if msg_type else ''
    command.append(tpl.format(msg, sound).encode('utf-8'))
    return command

def notify_command_linux(msg: str, duration_time: Optional[int] = None) -> List[Union[str, bytes]]:
    command = ['/usr/bin/notify-send']
    command.append(msg.encode('utf-8'))
    if duration_time:
        command.extend(['-t', str(duration_time)])
    command.extend(['-h', 'int:transient:1'])
    return command

def notify(msg: str, msg_type: int = 0, duration_time: Optional[int] = None) -> bool:
    """Show system notification with duration t (ms)"""
    msg = msg.replace('"', '\\"')
    if platform.system() == 'Darwin':
        command = notify_command_osx(msg, msg_type, duration_time)
    else:
        command = notify_command_linux(msg, duration_time)
    try:
        subprocess.call(command)
        return True
    except OSError:
        return False

if __name__ == '__main__':
    notify('I\'m test ""quote', msg_type=1, duration_time=1000)
    notify("I'm test 1", msg_type=1, duration_time=1000)
    notify("I'm test 2", msg_type=0, duration_time=1000)
