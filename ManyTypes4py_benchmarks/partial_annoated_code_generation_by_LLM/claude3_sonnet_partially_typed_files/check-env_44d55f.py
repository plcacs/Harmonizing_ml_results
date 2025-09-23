import platform
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import click
import psutil
from packaging.version import InvalidVersion, Version

class Requirement:

    def __init__(self, name: str, ideal_range: Tuple[Version, Version], supported_range: Tuple[Version, Version], req_type: str, command: str, version_post_process: Optional[Callable[[str], str]]=None):
        self.name: str = name
        self.ideal_range: Tuple[Version, Version] = ideal_range
        self.supported_range: Tuple[Version, Version] = supported_range
        self.req_type: str = req_type
        self.command: str = command
        self.version_post_process: Optional[Callable[[str], str]] = version_post_process
        self.version: Optional[str] = self.get_version()
        self.status: str = self.check_version()

    def get_version(self) -> Optional[str]:
        try:
            version: str = subprocess.check_output(self.command, shell=True).decode().strip()
            if self.version_post_process:
                version = self.version_post_process(version)
            return version.split()[-1]
        except subprocess.CalledProcessError:
            return None

    def check_version(self) -> str:
        if self.version is None:
            return '❌ Not Installed'
        try:
            version_number: Version = Version(self.version)
        except InvalidVersion:
            return '❌ Invalid Version Format'
        (ideal_min, ideal_max) = self.ideal_range
        (supported_min, supported_max) = self.supported_range
        if ideal_min <= version_number <= ideal_max:
            return '✅ Ideal'
        elif supported_min <= version_number:
            return '🟡 Supported'
        else:
            return '❌ Unsupported'

    def format_result(self) -> str:
        ideal_range_str: str = f'{self.ideal_range[0]} - {self.ideal_range[1]}'
        supported_range_str: str = f'{self.supported_range[0]} - {self.supported_range[1]}'
        return f"{self.status.split()[0]} {self.name:<25} {self.version or 'N/A':<25} {ideal_range_str:<25} {supported_range_str:<25}"

def check_memory(min_gb: float) -> str:
    total_memory: float = psutil.virtual_memory().total / 1024 ** 3
    if total_memory >= min_gb:
        return f'✅ Memory: {total_memory:.2f} GB'
    else:
        return f'❌ Memory: {total_memory:.2f} GB (Minimum required: {min_gb} GB)'

def get_cpu_info() -> str:
    cpu_count: int = psutil.cpu_count(logical=True)
    cpu_freq: Optional[psutil._common.scpufreq] = psutil.cpu_freq()
    cpu_info: str = f'{cpu_count} cores at {cpu_freq.current:.2f} MHz' if cpu_freq else f'{cpu_count} cores'
    return f'CPU: {cpu_info}'

def get_docker_platform() -> str:
    try:
        output: str = subprocess.check_output("docker info --format '{{.OperatingSystem}}'", shell=True).decode().strip()
        if 'Docker Desktop' in output:
            return f'Docker Platform: {output} ({platform.system()})'
        return f'Docker Platform: {output}'
    except subprocess.CalledProcessError:
        return 'Docker Platform: ❌ Not Detected'

@click.command(help='\nThis script checks the local environment for various software versions and other requirements, providing feedback on whether they are ideal, supported, or unsupported.\n')
@click.option('--docker', is_flag=True, help='Check Docker and Docker Compose requirements')
@click.option('--frontend', is_flag=True, help='Check frontend requirements (npm, Node.js, memory)')
@click.option('--backend', is_flag=True, help='Check backend requirements (Python)')
def main(docker: bool, frontend: bool, backend: bool) -> None:
    requirements: List[Requirement] = [Requirement('python', (Version('3.10.0'), Version('3.10.999')), (Version('3.9.0'), Version('3.11.999')), 'backend', 'python --version'), Requirement('npm', (Version('10.0.0'), Version('999.999.999')), (Version('10.0.0'), Version('999.999.999')), 'frontend', 'npm -v'), Requirement('node', (Version('20.0.0'), Version('20.999.999')), (Version('20.0.0'), Version('20.999.999')), 'frontend', 'node -v'), Requirement('docker', (Version('20.10.0'), Version('999.999.999')), (Version('19.0.0'), Version('999.999.999')), 'docker', 'docker --version', lambda v: v.split(',')[0]), Requirement('docker-compose', (Version('2.28.0'), Version('999.999.999')), (Version('1.29.0'), Version('999.999.999')), 'docker', 'docker-compose --version'), Requirement('git', (Version('2.30.0'), Version('999.999.999')), (Version('2.20.0'), Version('999.999.999')), 'backend', 'git --version')]
    print('==================')
    print('System Information')
    print('==================')
    print(f'OS: {platform.system()} {platform.release()}')
    print(get_cpu_info())
    print(get_docker_platform())
    print('\n')
    check_req_types: Set[str] = set()
    if docker:
        check_req_types.add('docker')
    if frontend:
        check_req_types.add('frontend')
    if backend:
        check_req_types.add('backend')
    if not check_req_types:
        check_req_types.update(['docker', 'frontend', 'backend'])
    headers: List[str] = ['Status', 'Software', 'Version Found', 'Ideal Range', 'Supported Range']
    row_format: str = '{:<2} {:<25} {:<25} {:<25} {:<25}'
    print('=' * 100)
    print(row_format.format(*headers))
    print('=' * 100)
    all_ok: bool = True
    for requirement in requirements:
        if requirement.req_type in check_req_types:
            result: str = requirement.format_result()
            if '❌' in requirement.status:
                all_ok = False
            print(result)
    if 'frontend' in check_req_types:
        memory_check: str = check_memory(12)
        if '❌' in memory_check:
            all_ok = False
        print(memory_check)
    if not all_ok:
        sys.exit(1)
if __name__ == '__main__':
    main()
