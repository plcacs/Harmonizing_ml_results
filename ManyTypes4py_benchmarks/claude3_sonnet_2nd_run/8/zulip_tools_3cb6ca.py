import argparse
import configparser
import functools
import hashlib
import json
import logging
import os
import pwd
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zoneinfo
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import IO, Any, Callable, Dict, List, Literal, Optional, Set, Tuple, TypeVar, Union, overload
from urllib.parse import SplitResult

DEPLOYMENTS_DIR: str = '/home/zulip/deployments'
LOCK_DIR: str = os.path.join(DEPLOYMENTS_DIR, 'lock')
TIMESTAMP_FORMAT: str = '%Y-%m-%d-%H-%M-%S'
OKBLUE: str = '\x1b[94m'
OKGREEN: str = '\x1b[92m'
WARNING: str = '\x1b[93m'
FAIL: str = '\x1b[91m'
ENDC: str = '\x1b[0m'
BLACKONYELLOW: str = '\x1b[0;30;43m'
WHITEONRED: str = '\x1b[0;37;41m'
BOLDRED: str = '\x1b[1;31m'
BOLD: str = '\x1b[1m'
GRAY: str = '\x1b[90m'
GREEN: str = '\x1b[32m'
YELLOW: str = '\x1b[33m'
BLUE: str = '\x1b[34m'
MAGENTA: str = '\x1b[35m'
CYAN: str = '\x1b[36m'

def overwrite_symlink(src: str, dst: str) -> None:
    dir, base = os.path.split(dst)
    while True:
        tmp = os.path.join(dir, f'.{base}.{random.randrange(1 << 40):010x}')
        try:
            os.symlink(src, tmp)
        except FileExistsError:
            continue
        break
    try:
        os.rename(tmp, dst)
    except BaseException:
        os.remove(tmp)
        raise

def parse_cache_script_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--threshold', dest='threshold_days', type=int, default=14, metavar='<days>', help='Any cache which is not in use by a deployment not older than threshold days(current installation in dev) and older than threshold days will be deleted. (defaults to 14)')
    parser.add_argument('--dry-run', action='store_true', help='If specified then script will only print the caches that it will delete/keep back. It will not delete any cache.')
    parser.add_argument('--verbose', action='store_true', help='If specified then script will print a detailed report of what is being will deleted/kept back.')
    parser.add_argument('--no-print-headings', dest='no_headings', action='store_true', help='If specified then script will not print headings for what will be deleted/kept back.')
    args = parser.parse_args()
    args.verbose |= args.dry_run
    return args

def get_deploy_root() -> str:
    return os.path.realpath(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))

def parse_version_from(deploy_path: str, merge_base: bool = False) -> str:
    if not os.path.exists(os.path.join(deploy_path, 'zulip-git-version')):
        try:
            subprocess.check_call([os.path.join(get_deploy_root(), 'scripts', 'lib', 'update-git-upstream')], cwd=deploy_path, preexec_fn=su_to_zulip)
            subprocess.check_call([os.path.join(deploy_path, 'tools', 'cache-zulip-git-version')], cwd=deploy_path, preexec_fn=su_to_zulip)
        except subprocess.CalledProcessError:
            pass
    try:
        varname = 'ZULIP_MERGE_BASE' if merge_base else 'ZULIP_VERSION'
        return subprocess.check_output([sys.executable, '-c', f'from version import {varname}; print({varname})'], cwd=deploy_path, text=True).strip()
    except subprocess.CalledProcessError:
        return '0.0.0'

def get_deployment_version(extract_path: str) -> str:
    version = '0.0.0'
    for item in os.listdir(extract_path):
        item_path = os.path.join(extract_path, item)
        if item.startswith('zulip-server') and os.path.isdir(item_path):
            version = parse_version_from(item_path)
            break
    return version

def is_invalid_upgrade(current_version: str, new_version: str) -> bool:
    if new_version > '1.4.3' and current_version <= '1.3.10':
        return True
    return False

def get_zulip_pwent() -> pwd.struct_passwd:
    deploy_root_uid = os.stat(get_deploy_root()).st_uid
    if deploy_root_uid != 0:
        return pwd.getpwuid(deploy_root_uid)
    return pwd.getpwnam('zulip')

def get_postgres_pwent() -> pwd.struct_passwd:
    try:
        return pwd.getpwnam('postgres')
    except KeyError:
        return get_zulip_pwent()

def su_to_zulip(save_suid: bool = False) -> None:
    """Warning: su_to_zulip assumes that the zulip checkout is owned by
    the zulip user (or whatever normal user is running the Zulip
    installation).  It should never be run from the installer or other
    production contexts before /home/zulip/deployments/current is
    created."""
    pwent = get_zulip_pwent()
    os.setgid(pwent.pw_gid)
    if save_suid:
        os.setresuid(pwent.pw_uid, pwent.pw_uid, os.getuid())
    else:
        os.setuid(pwent.pw_uid)
    os.environ['HOME'] = pwent.pw_dir

def make_deploy_path() -> str:
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    return os.path.join(DEPLOYMENTS_DIR, timestamp)

TEMPLATE_DATABASE_DIR: str = 'test-backend/databases'

def get_dev_uuid_var_path(create_if_missing: bool = False) -> str:
    zulip_path = get_deploy_root()
    uuid_path = os.path.join(os.path.realpath(os.path.dirname(zulip_path)), '.zulip-dev-uuid')
    if os.path.exists(uuid_path):
        with open(uuid_path) as f:
            zulip_uuid = f.read().strip()
    elif create_if_missing:
        zulip_uuid = str(uuid.uuid4())
        run_as_root(['sh', '-c', 'echo "$1" > "$2"', '-', zulip_uuid, uuid_path])
    else:
        raise AssertionError('Missing UUID file; please run tools/provision!')
    result_path = os.path.join(zulip_path, 'var', zulip_uuid)
    os.makedirs(result_path, exist_ok=True)
    return result_path

def get_deployment_lock(error_rerun_script: str) -> None:
    start_time = time.time()
    got_lock = False
    while time.time() - start_time < 300:
        try:
            os.mkdir(LOCK_DIR)
            got_lock = True
            break
        except OSError:
            print(WARNING + 'Another deployment in progress; waiting for lock... ' + f'(If no deployment is running, rmdir {LOCK_DIR})' + ENDC, flush=True)
            time.sleep(3)
    if not got_lock:
        print(FAIL + 'Deployment already in progress.  Please run\n' + f'  {error_rerun_script}\n' + 'manually when the previous deployment finishes, or run\n' + f'  rmdir {LOCK_DIR}\n' + 'if the previous deployment crashed.' + ENDC)
        sys.exit(1)

def release_deployment_lock() -> None:
    shutil.rmtree(LOCK_DIR)

def run(args: List[str], **kwargs: Any) -> None:
    print(f'+ {shlex.join(args)}', flush=True)
    try:
        subprocess.check_call(args, **kwargs)
    except subprocess.CalledProcessError as error:
        print()
        if error.returncode < 0:
            try:
                signal_name = signal.Signals(-error.returncode).name
            except ValueError:
                signal_name = f'unknown signal {-error.returncode}'
            print(WHITEONRED + f'Subcommand of {sys.argv[0]} died with {signal_name}: {shlex.join(args)}' + ENDC)
        else:
            print(WHITEONRED + f'Subcommand of {sys.argv[0]} failed with exit status {error.returncode}: {shlex.join(args)}' + ENDC)
            print(WHITEONRED + 'Actual error output for the subcommand is just above this.' + ENDC)
        print()
        sys.exit(1)

def log_management_command(cmd: List[str], log_path: str) -> None:
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('zulip.management')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info('Ran %s', shlex.join(cmd))

def get_environment() -> str:
    if os.path.exists(DEPLOYMENTS_DIR):
        return 'prod'
    return 'dev'

def get_recent_deployments(threshold_days: int) -> Set[str]:
    recent = set()
    threshold_date = datetime.now() - timedelta(days=threshold_days)
    for dir_name in os.listdir(DEPLOYMENTS_DIR):
        target_dir = os.path.join(DEPLOYMENTS_DIR, dir_name)
        if not os.path.isdir(target_dir):
            continue
        if not os.path.exists(os.path.join(target_dir, 'zerver')):
            continue
        try:
            date = datetime.strptime(dir_name, TIMESTAMP_FORMAT)
            if date >= threshold_date:
                recent.add(target_dir)
        except ValueError:
            recent.add(target_dir)
            if os.path.islink(target_dir):
                recent.add(os.path.realpath(target_dir))
    if os.path.exists('/root/zulip'):
        recent.add('/root/zulip')
    return recent

def get_threshold_timestamp(threshold_days: int) -> int:
    threshold = datetime.now() - timedelta(days=threshold_days)
    threshold_timestamp = int(time.mktime(threshold.utctimetuple()))
    return threshold_timestamp

def get_caches_to_be_purged(caches_dir: str, caches_in_use: Set[str], threshold_days: int) -> Set[str]:
    caches_to_purge = set()
    threshold_timestamp = get_threshold_timestamp(threshold_days)
    for cache_dir_base in os.listdir(caches_dir):
        cache_dir = os.path.join(caches_dir, cache_dir_base)
        if cache_dir in caches_in_use:
            continue
        if os.path.getctime(cache_dir) < threshold_timestamp:
            caches_to_purge.add(cache_dir)
    return caches_to_purge

def purge_unused_caches(caches_dir: str, caches_in_use: Set[str], cache_type: str, args: argparse.Namespace) -> None:
    if not os.path.exists(caches_dir):
        return
    all_caches = {os.path.join(caches_dir, cache) for cache in os.listdir(caches_dir)}
    caches_to_purge = get_caches_to_be_purged(caches_dir, caches_in_use, args.threshold_days)
    caches_to_keep = all_caches - caches_to_purge
    maybe_perform_purging(caches_to_purge, caches_to_keep, cache_type, args.dry_run, args.verbose, args.no_headings)
    if args.verbose:
        print('Done!')

def generate_sha1sum_emoji(zulip_path: str) -> str:
    sha = hashlib.sha1()
    filenames = ['web/images/zulip-emoji/zulip.png', 'tools/setup/emoji/emoji_map.json', 'tools/setup/emoji/build_emoji', 'tools/setup/emoji/emoji_setup_utils.py', 'tools/setup/emoji/emoji_names.py', 'zerver/management/data/unified_reactions.json']
    for filename in filenames:
        file_path = os.path.join(zulip_path, filename)
        with open(file_path, 'rb') as reader:
            sha.update(reader.read())
    with open(os.path.join(zulip_path, 'node_modules/emoji-datasource-google/package.json')) as fp:
        emoji_datasource_version = json.load(fp)['version']
    sha.update(emoji_datasource_version.encode())
    return sha.hexdigest()

def maybe_perform_purging(dirs_to_purge: Set[str], dirs_to_keep: Set[str], dir_type: str, dry_run: bool, verbose: bool, no_headings: bool) -> None:
    if dry_run:
        print('Performing a dry run...')
    if not no_headings:
        print(f'Cleaning unused {dir_type}s...')
    for directory in dirs_to_purge:
        if verbose:
            print(f'Cleaning unused {dir_type}: {directory}')
        if not dry_run:
            run_as_root(['rm', '-rf', directory])
    for directory in dirs_to_keep:
        if verbose:
            print(f'Keeping used {dir_type}: {directory}')

@functools.lru_cache(None)
def parse_os_release() -> Dict[str, str]:
    """
    Example of the useful subset of the data:
    {
     'ID': 'ubuntu',
     'VERSION_ID': '18.04',
     'NAME': 'Ubuntu',
     'VERSION': '18.04.3 LTS (Bionic Beaver)',
     'PRETTY_NAME': 'Ubuntu 18.04.3 LTS',
    }

    VERSION_CODENAME (e.g. 'bionic') is nice and readable to Ubuntu
    developers, but we avoid using it, as it is not available on
    RHEL-based platforms.
    """
    distro_info: Dict[str, str] = {}
    with open('/etc/os-release') as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            k, v = line.split('=', 1)
            [distro_info[k]] = shlex.split(v)
    return distro_info

@functools.lru_cache(None)
def os_families() -> Set[str]:
    """
    Known families:
    debian (includes: debian, ubuntu)
    ubuntu (includes: ubuntu)
    fedora (includes: fedora, rhel, centos)
    rhel (includes: rhel, centos)
    centos (includes: centos)
    """
    distro_info = parse_os_release()
    return {distro_info['ID'], *distro_info.get('ID_LIKE', '').split()}

def get_tzdata_zi() -> IO[str]:
    for path in zoneinfo.TZPATH:
        filename = os.path.join(path, 'tzdata.zi')
        if os.path.exists(filename):
            return open(filename)
    raise RuntimeError('Missing time zone data (tzdata.zi)')

def files_and_string_digest(filenames: List[str], extra_strings: List[str]) -> str:
    sha1sum = hashlib.sha1()
    for fn in filenames:
        with open(fn, 'rb') as file_to_hash:
            sha1sum.update(file_to_hash.read())
    for extra_string in extra_strings:
        sha1sum.update(extra_string.encode())
    return sha1sum.hexdigest()

def is_digest_obsolete(hash_name: str, filenames: List[str], extra_strings: List[str] = []) -> bool:
    """
    In order to determine if we need to run some
    process, we calculate a digest of the important
    files and strings whose respective contents
    or values may indicate such a need.

        filenames = files we should hash the contents of
        extra_strings = strings we should hash directly

    Grep for callers to see examples of how this is used.

    To elaborate on extra_strings, they will typically
    be things like:

        - package versions (that we import)
        - settings values (that we stringify with
          json, deterministically)
    """
    last_hash_path = os.path.join(get_dev_uuid_var_path(), hash_name)
    try:
        with open(last_hash_path) as f:
            old_hash = f.read()
    except FileNotFoundError:
        return True
    new_hash = files_and_string_digest(filenames, extra_strings)
    return new_hash != old_hash

def write_new_digest(hash_name: str, filenames: List[str], extra_strings: List[str] = []) -> None:
    hash_path = os.path.join(get_dev_uuid_var_path(), hash_name)
    new_hash = files_and_string_digest(filenames, extra_strings)
    with open(hash_path, 'w') as f:
        f.write(new_hash)
    print('New digest written to: ' + hash_path)

def is_root() -> bool:
    if 'posix' in os.name and os.geteuid() == 0:
        return True
    return False

def run_as_root(args: List[str], **kwargs: Any) -> None:
    sudo_args: List[str] = kwargs.pop('sudo_args', [])
    if not is_root():
        args = ['sudo', *sudo_args, '--', *args]
    run(args, **kwargs)

def assert_not_running_as_root() -> None:
    script_name = os.path.abspath(sys.argv[0])
    if is_root():
        pwent = get_zulip_pwent()
        msg = f"{os.path.basename(script_name)} should not be run as root. Use `su {pwent.pw_name}` to switch to the 'zulip'\nuser before rerunning this, or use \n  su {pwent.pw_name} -c '{script_name} ...'\nto switch users and run this as a single command."
        print(msg)
        sys.exit(1)

def assert_running_as_root(strip_lib_from_paths: bool = False) -> None:
    script_name = os.path.abspath(sys.argv[0])
    if strip_lib_from_paths:
        script_name = script_name.replace('scripts/lib/upgrade', 'scripts/upgrade')
    if not is_root():
        print(f'{script_name} must be run as root.')
        sys.exit(1)

@overload
def get_config(config_file: configparser.RawConfigParser, section: str, key: str, default_value: None = None) -> Optional[str]:
    ...

@overload
def get_config(config_file: configparser.RawConfigParser, section: str, key: str, default_value: bool) -> bool:
    ...

@overload
def get_config(config_file: configparser.RawConfigParser, section: str, key: str, default_value: str) -> str:
    ...

def get_config(config_file: configparser.RawConfigParser, section: str, key: str, default_value: Optional[Union[str, bool]] = None) -> Optional[Union[str, bool]]:
    if config_file.has_option(section, key):
        val = config_file.get(section, key)
        if isinstance(default_value, bool):
            return val.lower() in ['1', 'y', 't', 'true', 'yes', 'enable', 'enabled']
        return val
    return default_value

def get_config_file() -> configparser.RawConfigParser:
    config_file = configparser.RawConfigParser()
    config_file.read('/etc/zulip/zulip.conf')
    return config_file

def get_deploy_options(config_file: configparser.RawConfigParser) -> List[str]:
    return shlex.split(get_config(config_file, 'deployment', 'deploy_options', '') or '')

def run_psql_as_postgres(config_file: configparser.RawConfigParser, sql_query: str) -> None:
    dbname = get_config(config_file, 'postgresql', 'database_name', 'zulip')
    subcmd = shlex.join(['psql', '-v', 'ON_ERROR_STOP=1', '-d', dbname, '-c', sql_query])
    subprocess.check_call(['su', 'postgres', '-c', subcmd])

def get_tornado_ports(config_file: configparser.RawConfigParser) -> List[int]:
    ports: List[int] = []
    if config_file.has_section('tornado_sharding'):
        ports = sorted({int(port) for key in config_file.options('tornado_sharding') for port in key.removesuffix('_regex').split('_')})
    if not ports:
        ports = [9800]
    return ports

def get_or_create_dev_uuid_var_path(path: str) -> str:
    absolute_path = f'{get_dev_uuid_var_path()}/{path}'
    os.makedirs(absolute_path, exist_ok=True)
    return absolute_path

def is_vagrant_env_host(path: str) -> bool:
    return '.vagrant' in os.listdir(path)

def has_application_server(once: bool = False) -> bool:
    if once:
        return os.path.exists('/etc/supervisor/conf.d/zulip/zulip-once.conf')
    return os.path.exists('/etc/supervisor/conf.d/zulip/zulip.conf') or os.path.exists('/etc/supervisor/conf.d/zulip.conf')

def has_process_fts_updates() -> bool:
    return os.path.exists('/etc/supervisor/conf.d/zulip/zulip_db.conf') or os.path.exists('/etc/supervisor/conf.d/zulip_db.conf')

def deport(netloc: str) -> str:
    """Remove the port from a hostname:port string.  Brackets on a literal
    IPv6 address are included."""
    r = SplitResult('', netloc, '', '', '')
    assert r.hostname is not None
    return '[' + r.hostname + ']' if ':' in r.hostname else r.hostname

def start_arg_parser(action: str, add_help: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('--fill-cache', action='store_true', help='Fill the memcached caches')
    parser.add_argument('--skip-checks', action='store_true', help='Skip syntax and database checks')
    which_services = parser.add_mutually_exclusive_group()
    which_services.add_argument('--skip-client-reloads', action='store_true', help='Do not send reload events to web clients')
    which_services.add_argument('--only-django', action='store_true', help=f'Only {action} Django (not Tornado or workers)')
    if action == 'restart':
        parser.add_argument('--less-graceful', action='store_true', help='Restart with more concern for expediency than minimizing availability interruption')
    return parser

def listening_publicly(port: int) -> List[str]:
    filter = f'sport = :{port} and not src 127.0.0.1:{port} and not src [::1]:{port}'
    lines = subprocess.check_output(['/bin/ss', '-Hnl', filter], text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    return [line.split()[4] for line in lines]

def atomic_nagios_write(name: str, status: str, message: Optional[str] = None, event_time: Optional[int] = None) -> int:
    if message is None:
        message = status
    if event_time is None:
        event_time = int(time.time())
    if status == 'ok':
        status_int = 0
    elif status == 'warning':
        status_int = 1
    elif status == 'critical':
        status_int = 2
    elif status == 'unknown':
        status_int = 3
    path = '/var/lib/nagios_state/' + name
    with open(path + '.tmp', 'w') as fh:
        fh.write('|'.join([str(event_time), str(status_int), status, message]) + '\n')
    os.rename(path + '.tmp', path)
    return status_int

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'make_deploy_path':
        print(make_deploy_path())
    elif cmd == 'get_dev_uuid':
        print(get_dev_uuid_var_path())
