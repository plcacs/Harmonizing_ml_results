import os
import sys
from copy import copy
from tempfile import mkdtemp
from typing import Callable, Dict, Iterable, List, Optional, Any, ContextManager
import pexpect
import pytest
from raiden.constants import MATRIX_AUTO_SELECT_SERVER, Environment, EthClient
from raiden.settings import RAIDEN_CONTRACT_VERSION
from raiden.tests.utils.ci import get_artifacts_storage
from raiden.tests.utils.smoketest import setup_raiden, setup_testchain
from raiden.tests.utils.transport import ParsedURL
from raiden.utils.typing import Any, ContextManager, Dict, List, Optional

@pytest.fixture(scope='module')
def cli_tests_contracts_version() -> str:
    return RAIDEN_CONTRACT_VERSION

@pytest.fixture(scope='module')
def raiden_testchain(blockchain_type: str, port_generator: Iterable[int], cli_tests_contracts_version: str) -> Dict[str, Any]:
    import time
    start_time = time.monotonic()
    eth_client = EthClient(blockchain_type)
    tmpdir = mkdtemp()
    base_datadir = str(tmpdir)
    base_logdir = os.path.join(get_artifacts_storage() or str(tmpdir), blockchain_type)
    os.makedirs(base_logdir, exist_ok=True)
    testchain_manager = setup_testchain(eth_client=eth_client, free_port_generator=port_generator, base_datadir=base_datadir, base_logdir=base_logdir)

    def dont_print_step(description: str, error: bool = False) -> None:
        pass
    with testchain_manager as testchain:
        result = setup_raiden(matrix_server=MATRIX_AUTO_SELECT_SERVER, print_step=dont_print_step, contracts_version=cli_tests_contracts_version, eth_rpc_endpoint=testchain['eth_rpc_endpoint'], web3=testchain['web3'], base_datadir=testchain['base_datadir'], keystore=testchain['keystore'], free_port_generator=port_generator)
        args = result.args
        print('setup_raiden took', time.monotonic() - start_time)
        yield args

@pytest.fixture()
def removed_args() -> Optional[List[str]]:
    return None

@pytest.fixture()
def changed_args() -> Optional[Dict[str, Any]]:
    return None

@pytest.fixture()
def cli_args(logs_storage: str, raiden_testchain: Dict[str, Any], local_matrix_servers: List[str], removed_args: Optional[List[str]], changed_args: Optional[Dict[str, Any]], environment_type: Environment) -> List[str]:
    initial_args = raiden_testchain.copy()
    if removed_args is not None:
        for arg in removed_args:
            if arg in initial_args:
                del initial_args[arg]
    if changed_args is not None:
        for k, v in changed_args.items():
            initial_args[k] = v
    initial_args['network_id'] = initial_args.pop('chain_id')
    base_logfile = os.path.join(logs_storage, 'raiden_nodes', 'cli_test.log')
    os.makedirs(os.path.dirname(base_logfile), exist_ok=True)
    args = ['--gas-price', '1000000000', '--no-sync-check', f'--debug-logfile-path={base_logfile}', '--matrix-server', local_matrix_servers[0]]
    args += ['--environment-type', environment_type.value]
    for arg_name, arg_value in initial_args.items():
        if arg_name == 'sync_check':
            continue
        arg_name_cli = '--' + arg_name.replace('_', '-')
        if arg_name_cli not in args:
            args.append(arg_name_cli)
            if arg_value is not None:
                args.append(arg_value)
    return args

@pytest.fixture
def raiden_spawner(tmp_path: Any, request: Any) -> Callable[[List[str]], pexpect.spawn]:

    def spawn_raiden(args: List[str]) -> pexpect.spawn:
        new_env = {k: copy(v) for k, v in os.environ.items() if not k.startswith('RAIDEN')}
        new_env['HOME'] = str(tmp_path)
        child = pexpect.spawn(sys.executable, ['-m', 'raiden'] + args, logfile=sys.stdout, encoding='utf-8', env=new_env, timeout=None)
        request.addfinalizer(child.close)
        return child
    return spawn_raiden
