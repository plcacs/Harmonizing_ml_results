from typing import NamedTuple, Sequence, Any
from eth_utils import encode_hex
from eth2spec.test.exceptions import BlockNotFoundException
from eth2spec.test.helpers.attestations import next_epoch_with_attestations, next_slots_with_attestations, state_transition_with_full_block


class BlobData(NamedTuple):
    """
    The return values of ``retrieve_blobs_and_proofs`` helper.
    """


def func_87ns3l7o(spec, blob_data, func):
    """
    This helper runs the given ``func`` with monkeypatched ``retrieve_blobs_and_proofs``
    that returns ``blob_data.blobs, blob_data.proofs``.
    """

    def func_226uwqj8(beacon_block_root):
        return blob_data.blobs, blob_data.proofs
    retrieve_blobs_and_proofs_backup = spec.retrieve_blobs_and_proofs
    spec.retrieve_blobs_and_proofs = retrieve_blobs_and_proofs


    class AtomicBoolean:
        value = False
    is_called = AtomicBoolean()

    def func_zf79nsgl(flag):
        yield from func()
        flag.value = True
    try:
        yield from func_zf79nsgl(is_called)
    finally:
        spec.retrieve_blobs_and_proofs = retrieve_blobs_and_proofs_backup
    assert is_called.value


def func_hv70aeh2(spec, state):
    anchor_block_header = state.latest_block_header.copy()
    if anchor_block_header.state_root == spec.Bytes32():
        anchor_block_header.state_root = spec.hash_tree_root(state)
    return spec.hash_tree_root(anchor_block_header)


def func_6nd9w86x(spec, store, signed_block, test_steps, valid=True,
    merge_block=False, block_not_found=False, is_optimistic=False,
    blob_data=None):
    pre_state = store.block_states[signed_block.message.parent_root]
    if merge_block:
        assert spec.is_merge_transition_block(pre_state, signed_block.
            message.body)
    block_time = (pre_state.genesis_time + signed_block.message.slot * spec
        .config.SECONDS_PER_SLOT)
    while store.time < block_time:
        time = pre_state.genesis_time + (spec.get_current_slot(store) + 1
            ) * spec.config.SECONDS_PER_SLOT
        on_tick_and_append_step(spec, store, time, test_steps)
    post_state = yield from add_block(spec, store, signed_block, test_steps,
        valid=valid, block_not_found=block_not_found, is_optimistic=
        is_optimistic, blob_data=blob_data)
    return post_state


def func_59mbatyk(spec, store, signed_block, test_steps, blob_data, valid=True
    ):

    def func_9vowawh7():
        yield from func_6nd9w86x(spec, store, signed_block, test_steps,
            blob_data=blob_data, valid=valid)
    yield from func_87ns3l7o(spec, blob_data, run_func)


def func_zhxryae2(spec, store, attestation, test_steps, is_from_block=False):
    spec.on_attestation(store, attestation, is_from_block=is_from_block)
    yield get_attestation_file_name(attestation), attestation
    test_steps.append({'attestation': get_attestation_file_name(attestation)})


def func_9n3olnp9(spec, store, attestations, test_steps, is_from_block=False):
    for attestation in attestations:
        yield from func_zhxryae2(spec, store, attestation, test_steps,
            is_from_block=is_from_block)


def func_8ibosg19(spec, store, attestation, test_steps, is_from_block=False):
    min_time_to_include = (attestation.data.slot + 1
        ) * spec.config.SECONDS_PER_SLOT
    if store.time < min_time_to_include:
        spec.on_tick(store, min_time_to_include)
        test_steps.append({'tick': int(min_time_to_include)})
    yield from func_zhxryae2(spec, store, attestation, test_steps,
        is_from_block)


def func_zew13chv(spec, store, attestation, is_from_block=False, valid=True):
    if not valid:
        try:
            spec.on_attestation(store, attestation, is_from_block=is_from_block
                )
        except AssertionError:
            return
        else:
            assert False
    spec.on_attestation(store, attestation, is_from_block=is_from_block)


def func_98hjkg2u(spec, genesis_state):
    store, _ = get_genesis_forkchoice_store_and_block(spec, genesis_state)
    return store


def func_u1sbnzvr(spec, genesis_state):
    assert genesis_state.slot == spec.GENESIS_SLOT
    genesis_block = spec.BeaconBlock(state_root=genesis_state.hash_tree_root())
    return spec.get_forkchoice_store(genesis_state, genesis_block
        ), genesis_block


def func_2irvtzjy(block):
    return f'block_{encode_hex(block.hash_tree_root())}'


def func_q85npvr6(attestation):
    return f'attestation_{encode_hex(attestation.hash_tree_root())}'


def func_quhlfiys(attester_slashing):
    return (
        f'attester_slashing_{encode_hex(attester_slashing.hash_tree_root())}')


def func_bs2yh74y(blobs=None, blobs_root=None):
    if blobs:
        return f'blobs_{encode_hex(blobs.hash_tree_root())}'
    else:
        return f'blobs_{encode_hex(blobs_root)}'


def func_hpsjsp65(spec, store, time, test_steps):
    assert time >= store.time
    spec.on_tick(store, time)
    test_steps.append({'tick': int(time)})
    output_store_checks(spec, store, test_steps)


def func_v1vtdo7e(spec, store, signed_block, valid=True):
    if not valid:
        try:
            spec.on_block(store, signed_block)
        except AssertionError:
            return
        else:
            assert False
    spec.on_block(store, signed_block)
    assert store.blocks[signed_block.message.hash_tree_root()
        ] == signed_block.message


def func_yjhynh7e(spec, store, signed_block, test_steps, valid=True,
    block_not_found=False, is_optimistic=False, blob_data=None):
    """
    Run on_block and on_attestation
    """
    yield func_2irvtzjy(signed_block), signed_block
    if blob_data is not None:
        blobs = spec.List[spec.Blob, spec.MAX_BLOB_COMMITMENTS_PER_BLOCK](
            blob_data.blobs)
        blobs_root = blobs.hash_tree_root()
        yield func_bs2yh74y(blobs_root=blobs_root), blobs
    is_blob_data_test = blob_data is not None

    def func_5cwzuw3x(is_blob_data_test, valid=True):
        if is_blob_data_test:
            test_steps.append({'block': func_2irvtzjy(signed_block),
                'blobs': func_bs2yh74y(blobs_root=blobs_root), 'proofs': [
                encode_hex(proof) for proof in blob_data.proofs], 'valid':
                valid})
        else:
            test_steps.append({'block': func_2irvtzjy(signed_block),
                'valid': valid})
    if not valid:
        if is_optimistic:
            func_v1vtdo7e(spec, store, signed_block, valid=True)
            func_5cwzuw3x(is_blob_data_test, valid=False)
        else:
            try:
                func_v1vtdo7e(spec, store, signed_block, valid=True)
            except (AssertionError, BlockNotFoundException) as e:
                if isinstance(e, BlockNotFoundException
                    ) and not block_not_found:
                    assert False
                func_5cwzuw3x(is_blob_data_test, valid=False)
                return
            else:
                assert False
    else:
        func_v1vtdo7e(spec, store, signed_block, valid=True)
        func_5cwzuw3x(is_blob_data_test)
    for attestation in signed_block.message.body.attestations:
        func_zew13chv(spec, store, attestation, is_from_block=True, valid=True)
    for attester_slashing in signed_block.message.body.attester_slashings:
        run_on_attester_slashing(spec, store, attester_slashing, valid=True)
    block_root = signed_block.message.hash_tree_root()
    assert store.blocks[block_root] == signed_block.message
    assert store.block_states[block_root].hash_tree_root(
        ) == signed_block.message.state_root
    if not is_optimistic:
        output_store_checks(spec, store, test_steps)
    return store.block_states[signed_block.message.hash_tree_root()]


def func_oi6yrwsp(spec, store, attester_slashing, valid=True):
    if not valid:
        try:
            spec.on_attester_slashing(store, attester_slashing)
        except AssertionError:
            return
        else:
            assert False
    spec.on_attester_slashing(store, attester_slashing)


def func_aq614qdm(spec, store, attester_slashing, test_steps, valid=True):
    slashing_file_name = func_quhlfiys(attester_slashing)
    yield func_quhlfiys(attester_slashing), attester_slashing
    if not valid:
        try:
            func_oi6yrwsp(spec, store, attester_slashing)
        except AssertionError:
            test_steps.append({'attester_slashing': slashing_file_name,
                'valid': False})
            return
        else:
            assert False
    func_oi6yrwsp(spec, store, attester_slashing)
    test_steps.append({'attester_slashing': slashing_file_name})


def func_6l5hx250(spec, store):
    head = spec.get_head(store)
    slot = store.blocks[head].slot
    return {'slot': int(slot), 'root': encode_hex(head)}


def func_ejp5lw0x(spec, store, test_steps):
    test_steps.append({'checks': {'head': func_6l5hx250(spec, store)}})


def func_xn49xwcl(spec, store, test_steps):
    test_steps.append({'checks': {'time': int(store.time), 'head':
        func_6l5hx250(spec, store), 'justified_checkpoint': {'epoch': int(
        store.justified_checkpoint.epoch), 'root': encode_hex(store.
        justified_checkpoint.root)}, 'finalized_checkpoint': {'epoch': int(
        store.finalized_checkpoint.epoch), 'root': encode_hex(store.
        finalized_checkpoint.root)}, 'proposer_boost_root': encode_hex(
        store.proposer_boost_root)}})


def func_w35odo4x(spec, state, store, fill_cur_epoch, fill_prev_epoch,
    participation_fn=None, test_steps=None):
    if test_steps is None:
        test_steps = []
    _, new_signed_blocks, post_state = next_epoch_with_attestations(spec,
        state, fill_cur_epoch, fill_prev_epoch, participation_fn=
        participation_fn)
    for signed_block in new_signed_blocks:
        block = signed_block.message
        yield from func_6nd9w86x(spec, store, signed_block, test_steps)
        block_root = block.hash_tree_root()
        assert store.blocks[block_root] == block
        last_signed_block = signed_block
    assert store.block_states[block_root].hash_tree_root(
        ) == post_state.hash_tree_root()
    return post_state, store, last_signed_block


def func_rfi6lmfh(spec, state, store, slots, fill_cur_epoch,
    fill_prev_epoch, test_steps, participation_fn=None):
    _, new_signed_blocks, post_state = next_slots_with_attestations(spec,
        state, slots, fill_cur_epoch, fill_prev_epoch, participation_fn=
        participation_fn)
    for signed_block in new_signed_blocks:
        block = signed_block.message
        yield from func_6nd9w86x(spec, store, signed_block, test_steps)
        block_root = block.hash_tree_root()
        assert store.blocks[block_root] == block
        last_signed_block = signed_block
    assert store.block_states[block_root].hash_tree_root(
        ) == post_state.hash_tree_root()
    return post_state, store, last_signed_block


def func_tzz2658r(spec, state):
    """
    Check if the given ``state`` will trigger justification updates at epoch boundary.
    """
    temp_state = state.copy()
    spec.process_justification_and_finalization(temp_state)
    return (temp_state.current_justified_checkpoint.epoch > state.
        current_justified_checkpoint.epoch)


def func_e0oz0iwz(spec, state, fill_cur_epoch, fill_prev_epoch,
    participation_fn=None):
    temp_state = state.copy()
    signed_blocks = []
    justifying_slot = None
    while justifying_slot is None:
        signed_block = state_transition_with_full_block(spec, temp_state,
            fill_cur_epoch, fill_prev_epoch, participation_fn)
        signed_blocks.append(signed_block)
        if func_tzz2658r(spec, temp_state):
            justifying_slot = temp_state.slot
    return signed_blocks, justifying_slot


def func_hgbjgjmk(pow_block):
    return f'pow_block_{encode_hex(pow_block.block_hash)}'


def func_m8o1umub(spec, store, pow_block, test_steps):
    yield func_hgbjgjmk(pow_block), pow_block
    test_steps.append({'pow_block': func_hgbjgjmk(pow_block)})
