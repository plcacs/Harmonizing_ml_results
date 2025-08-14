def initiator_init(
    raiden: "RaidenService",
    transfer_identifier: PaymentID,
    transfer_amount: PaymentAmount,
    transfer_secret: Secret,
    transfer_secrethash: SecretHash,
    token_network_address: TokenNetworkAddress,
    target_address: TargetAddress,
    lock_timeout: Optional[BlockTimeout] = None,
    route_states: Optional[List[RouteState]] = None,
) -> Tuple[Optional[str], ActionInitInitiator]:
    transfer_state = TransferDescriptionWithSecretState(
        token_network_registry_address=raiden.default_registry.address,
        payment_identifier=transfer_identifier,
        amount=transfer_amount,
        token_network_address=token_network_address,
        initiator=InitiatorAddress(raiden.address),
        target=target_address,
        secret=transfer_secret,
        secrethash=transfer_secrethash,
        lock_timeout=lock_timeout,
    )

    error_msg: Optional[str] = None
    if route_states is None:
        our_address_metadata = raiden.transport.address_metadata

        msg = "Transport is not initialized with raiden-service"
        assert our_address_metadata is not None, msg

        error_msg, route_states, feedback_token = routing.get_best_routes(
            chain_state=views.state_from_raiden(raiden),
            token_network_address=token_network_address,
            one_to_n_address=raiden.default_one_to_n_address,
            from_address=InitiatorAddress(raiden.address),
            to_address=target_address,
            amount=transfer_amount,
            previous_address=None,
            privkey=raiden.privkey,
            our_address_metadata=our_address_metadata,
            pfs_proxy=raiden.pfs_proxy,
        )

        # Only prepare feedback when token is available
        if feedback_token is not None:
            for route_state in route_states:
                raiden.route_to_feedback_token[tuple(route_state.route)] = feedback_token

    return error_msg, ActionInitInitiator(transfer_state, route_states)
