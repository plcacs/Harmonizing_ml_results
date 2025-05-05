import pytest
from eth_utils import ValidationError, to_wei
from eth.chains.base import MiningChain
from eth.tools.builder.chain import (
    at_block_number,
    berlin_at,
    build,
    byzantium_at,
    constantinople_at,
    disable_pow_check,
    frontier_at,
    genesis,
    homestead_at,
    london_at,
    mine_block,
    mine_blocks,
    petersburg_at,
    spurious_dragon_at,
    tangerine_whistle_at,
)
from eth.tools.factories.transaction import new_dynamic_fee_transaction
from typing import Callable, Any, Tuple


@pytest.mark.parametrize(
    'vm_fn, miner_1_balance, miner_2_balance',
    (
        (frontier_at, 15.15625, 4.375),
        (homestead_at, 15.15625, 4.375),
        (tangerine_whistle_at, 15.15625, 4.375),
        (spurious_dragon_at, 15.15625, 4.375),
        (byzantium_at, 9.09375, 2.625),
        (constantinople_at, 6.0625, 1.75),
        (petersburg_at, 6.0625, 1.75),
    ),
)
def test_rewards(
    vm_fn: Callable[[int], Any],
    miner_1_balance: float,
    miner_2_balance: float,
) -> None:
    OTHER_MINER_ADDRESS: bytes = 20 * b'\x01'
    TOTAL_BLOCKS_CANONICAL_CHAIN: int = 3
    chain: MiningChain = build(
        MiningChain,
        vm_fn(0),
        disable_pow_check(),
        genesis(),
        mine_block(),
        mine_block(),
    )
    fork_chain: MiningChain = build(
        chain,
        at_block_number(1),
        mine_block(extra_data=b'fork-it!', coinbase=OTHER_MINER_ADDRESS),
    )
    uncle = fork_chain.get_block_header_by_hash(fork_chain.header.parent_hash)
    assert uncle.block_number == 2
    assert uncle != chain.get_canonical_head()
    chain = build(chain, mine_block(uncles=[uncle]))
    header = chain.get_canonical_head()
    block = chain.get_block_by_hash(header.hash)
    assert len(block.uncles) == 1
    assert block.uncles[0] == uncle
    vm = chain.get_vm()
    coinbase_balance: int = vm.state.get_balance(block.header.coinbase)
    other_miner_balance_val: int = vm.state.get_balance(uncle.coinbase)
    assert (
        coinbase_balance
        == vm.get_block_reward() * TOTAL_BLOCKS_CANONICAL_CHAIN + vm.get_nephew_reward()
    )
    assert other_miner_balance_val == vm.get_uncle_reward(block.number, uncle)
    assert coinbase_balance == to_wei(miner_1_balance, 'ether')
    assert other_miner_balance_val == to_wei(miner_2_balance, 'ether')


@pytest.mark.parametrize(
    'vm_fn, fork_at_block_number, miner_1_balance, miner_2_balance',
    (
        (frontier_at, 3, 50.15625, 1.25),
        (frontier_at, 4, 50.15625, 1.875),
        (frontier_at, 5, 50.15625, 2.5),
        (frontier_at, 6, 50.15625, 3.125),
        (frontier_at, 7, 50.15625, 3.75),
        (frontier_at, 8, 50.15625, 4.375),
        (homestead_at, 3, 50.15625, 1.25),
        (homestead_at, 4, 50.15625, 1.875),
        (homestead_at, 5, 50.15625, 2.5),
        (homestead_at, 6, 50.15625, 3.125),
        (homestead_at, 7, 50.15625, 3.75),
        (homestead_at, 8, 50.15625, 4.375),
        (tangerine_whistle_at, 3, 50.15625, 1.25),
        (tangerine_whistle_at, 4, 50.15625, 1.875),
        (tangerine_whistle_at, 5, 50.15625, 2.5),
        (tangerine_whistle_at, 6, 50.15625, 3.125),
        (tangerine_whistle_at, 7, 50.15625, 3.75),
        (tangerine_whistle_at, 8, 50.15625, 4.375),
        (spurious_dragon_at, 3, 50.15625, 1.25),
        (spurious_dragon_at, 4, 50.15625, 1.875),
        (spurious_dragon_at, 5, 50.15625, 2.5),
        (spurious_dragon_at, 6, 50.15625, 3.125),
        (spurious_dragon_at, 7, 50.15625, 3.75),
        (spurious_dragon_at, 8, 50.15625, 4.375),
        (byzantium_at, 3, 30.09375, 0.75),
        (byzantium_at, 4, 30.09375, 1.125),
        (byzantium_at, 5, 30.09375, 1.5),
        (byzantium_at, 6, 30.09375, 1.875),
        (byzantium_at, 7, 30.09375, 2.25),
        (byzantium_at, 8, 30.09375, 2.625),
        (constantinople_at, 3, 20.0625, 0.5),
        (constantinople_at, 4, 20.0625, 0.75),
        (constantinople_at, 5, 20.0625, 1),
        (constantinople_at, 6, 20.0625, 1.25),
        (constantinople_at, 7, 20.0625, 1.5),
        (constantinople_at, 8, 20.0625, 1.75),
        (petersburg_at, 3, 20.0625, 0.5),
        (petersburg_at, 4, 20.0625, 0.75),
        (petersburg_at, 5, 20.0625, 1),
        (petersburg_at, 6, 20.0625, 1.25),
        (petersburg_at, 7, 20.0625, 1.5),
        (petersburg_at, 8, 20.0625, 1.75),
    ),
)
def test_rewards_uncle_created_at_different_generations(
    vm_fn: Callable[[int], Any],
    fork_at_block_number: int,
    miner_1_balance: float,
    miner_2_balance: float,
) -> None:
    OTHER_MINER_ADDRESS: bytes = 20 * b'\x01'
    TOTAL_BLOCKS_CANONICAL_CHAIN: int = 10
    chain: MiningChain = build(
        MiningChain,
        vm_fn(0),
        disable_pow_check(),
        genesis(),
        mine_blocks(TOTAL_BLOCKS_CANONICAL_CHAIN - 1),
    )
    fork_chain: MiningChain = build(
        chain,
        at_block_number(fork_at_block_number),
        mine_block(extra_data=b'fork-it!', coinbase=OTHER_MINER_ADDRESS),
    )
    uncle = fork_chain.get_block_header_by_hash(fork_chain.header.parent_hash)
    assert uncle.block_number == fork_at_block_number + 1
    chain = build(chain, mine_block(uncles=[uncle]))
    header = chain.get_canonical_head()
    block = chain.get_block_by_hash(header.hash)
    vm = chain.get_vm()
    coinbase_balance: int = vm.state.get_balance(block.header.coinbase)
    other_miner_balance_val: int = vm.state.get_balance(uncle.coinbase)
    assert (
        coinbase_balance
        == vm.get_block_reward() * TOTAL_BLOCKS_CANONICAL_CHAIN + vm.get_nephew_reward()
    )
    assert other_miner_balance_val == vm.get_uncle_reward(block.number, uncle)
    assert coinbase_balance == to_wei(miner_1_balance, 'ether')
    assert other_miner_balance_val == to_wei(miner_2_balance, 'ether')


@pytest.mark.parametrize(
    'vm_fn',
    (
        frontier_at,
        homestead_at,
        tangerine_whistle_at,
        spurious_dragon_at,
        byzantium_at,
        constantinople_at,
        petersburg_at,
    ),
)
def test_uncle_block_inclusion_validity(vm_fn: Callable[[int], Any]) -> None:
    OTHER_MINER_ADDRESS: bytes = 20 * b'\x01'
    chain: MiningChain = build(
        MiningChain,
        vm_fn(0),
        disable_pow_check(),
        genesis(),
        mine_block(),
        mine_block(),
    )
    fork_chain: MiningChain = build(
        chain,
        at_block_number(1),
        mine_block(extra_data=b'fork-it!', coinbase=OTHER_MINER_ADDRESS),
    )
    uncle = fork_chain.get_block_header_by_hash(fork_chain.header.parent_hash)
    assert uncle.block_number == 2
    chain = build(chain, mine_blocks(6))
    with pytest.raises(ValidationError):
        chain = build(chain, mine_block(uncles=[uncle]))


@pytest.mark.parametrize(
    'vm_fn_uncle, vm_fn_nephew, miner_1_balance, miner_2_balance',
    (
        (frontier_at, homestead_at, 50.15625, 1.25),
        (homestead_at, tangerine_whistle_at, 50.15625, 1.25),
        (tangerine_whistle_at, spurious_dragon_at, 50.15625, 1.25),
        (spurious_dragon_at, byzantium_at, 36.09375, 0.75),
        (byzantium_at, constantinople_at, 23.0625, 0.5),
        (byzantium_at, petersburg_at, 23.0625, 0.5),
    ),
)
def test_rewards_nephew_uncle_different_vm(
    vm_fn_uncle: Callable[[int], Any],
    vm_fn_nephew: Callable[[int], Any],
    miner_1_balance: float,
    miner_2_balance: float,
) -> None:
    OTHER_MINER_ADDRESS: bytes = 20 * b'\x01'
    TOTAL_BLOCKS_CANONICAL_CHAIN: int = 10
    VM_CHANGE_BLOCK_NUMBER: int = 4
    chain: MiningChain = build(
        MiningChain,
        vm_fn_uncle(0),
        vm_fn_nephew(VM_CHANGE_BLOCK_NUMBER),
        disable_pow_check(),
        genesis(),
        mine_blocks(TOTAL_BLOCKS_CANONICAL_CHAIN - 1),
    )
    fork_chain: MiningChain = build(
        chain,
        at_block_number(3),
        mine_block(extra_data=b'fork-it!', coinbase=OTHER_MINER_ADDRESS),
    )
    uncle = fork_chain.get_block_header_by_hash(fork_chain.header.parent_hash)
    assert uncle.block_number == 4
    chain = build(chain, mine_block(uncles=[uncle]))
    header = chain.get_canonical_head()
    block = chain.get_block_by_hash(header.hash)
    assert len(block.uncles) == 1
    assert block.uncles[0] == uncle
    vm = chain.get_vm()
    coinbase_balance: int = vm.state.get_balance(block.header.coinbase)
    other_miner_balance_val: int = vm.state.get_balance(uncle.coinbase)
    uncle_vm = chain.get_vm_class_for_block_number(0)
    nephew_vm = chain.get_vm_class_for_block_number(VM_CHANGE_BLOCK_NUMBER)
    assert (
        coinbase_balance
        == uncle_vm.get_block_reward() * 3
        + nephew_vm.get_block_reward() * (TOTAL_BLOCKS_CANONICAL_CHAIN - 3)
        + vm.get_nephew_reward()
    )
    assert other_miner_balance_val == vm.get_uncle_reward(block.number, uncle)
    assert coinbase_balance == to_wei(miner_1_balance, 'ether')
    assert other_miner_balance_val == to_wei(miner_2_balance, 'ether')


@pytest.mark.parametrize(
    'max_total_price, max_priority_price, expected_miner_tips',
    (
        (10**9, 1, 0),
        (10**9 + 1, 2, 21000),
        (10**9 + 1, 1, 21000),
        (10**9 + 2, 1, 21000),
    ),
)
def test_eip1559_txn_rewards(
    max_total_price: int,
    max_priority_price: int,
    expected_miner_tips: int,
    funded_address: bytes,
    funded_address_private_key: Any,
) -> None:
    chain: MiningChain = build(
        MiningChain,
        berlin_at(0),
        london_at(1),
        disable_pow_check(),
        genesis(
            params=dict(gas_limit=10**7),
            state={funded_address: dict(balance=10**20)},
        ),
    )
    vm = chain.get_vm()
    txn = new_dynamic_fee_transaction(
        vm,
        from_=funded_address,
        to=funded_address,
        private_key=funded_address_private_key,
        max_priority_fee_per_gas=max_priority_price,
        max_fee_per_gas=max_total_price,
    )
    MINER: bytes = b'\x0f' * 20
    original_balance: int = vm.state.get_balance(MINER)
    chain.mine_all([txn], coinbase=MINER)
    new_balance: int = chain.get_vm().state.get_balance(MINER)
    BLOCK_REWARD: int = 2 * 10**18
    assert original_balance + BLOCK_REWARD + expected_miner_tips == new_balance
