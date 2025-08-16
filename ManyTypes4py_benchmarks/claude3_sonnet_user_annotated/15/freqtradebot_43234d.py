def handle_similar_open_order(
    self, trade: Trade, price: float, amount: float, side: str
) -> bool:
    """
    Keep existing open order if same amount and side otherwise cancel
    :param trade: Trade object of the trade we're analyzing
    :param price: Limit price of the potential new order
    :param amount: Quantity of assets of the potential new order
    :param side: Side of the potential new order
    :return: True if an existing similar order was found
    """
    if trade.has_open_orders:
        oo: Order | None = trade.select_order(side, True)
        if oo is not None:
            if (price == oo.price) and (side == oo.side) and (amount == oo.amount):
                logger.info(
                    f"A similar open order was found for {trade.pair}. "
                    f"Keeping existing {trade.exit_side} order. {price=},  {amount=}"
                )
                return True
        # cancel open orders of this trade if order is different
        self.cancel_open_orders_of_trade(
            trade,
            [trade.entry_side, trade.exit_side],
            constants.CANCEL_REASON["REPLACE"],
            True,
        )
        Trade.commit()
        return False

    return False
