import random
from uuid import UUID
from prefect import flow
from prefect.logging import get_run_logger
from prefect.input import RunInput

class NumberData(RunInput):
    number: int


@flow
async def sender_flow(receiver_flow_run_id: UUID):
    logger = get_run_logger()

    the_number = random.randint(1, 100)

    await NumberData(number=the_number).send_to(receiver_flow_run_id)

    receiver = NumberData.receive(flow_run_id=receiver_flow_run_id)
    squared = await receiver.next()

    logger.info(f"{the_number} squared is {squared.number}")
