    from typing import Annotated

    from fastapi import FastAPI, Path

    app = FastAPI()


    @app.get("/items/{item_id}")
    async def read_items(
        item_id: Annotated[int, Path(title="The ID of the item to get")],
    ):
        return {"item_id": item_id}
    