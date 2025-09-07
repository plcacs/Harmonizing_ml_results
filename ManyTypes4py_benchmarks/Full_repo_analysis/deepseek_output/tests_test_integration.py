import pytest
import sqlalchemy
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from databases import Database, DatabaseURL
from typing import Any, Dict, Generator, List
from tests.test_databases import DATABASE_URLS
from starlette.requests import Request

metadata: sqlalchemy.MetaData = sqlalchemy.MetaData()
notes: sqlalchemy.Table = sqlalchemy.Table(
    'notes', 
    metadata, 
    sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), 
    sqlalchemy.Column('text', sqlalchemy.String(length=100)), 
    sqlalchemy.Column('completed', sqlalchemy.Boolean)
)

@pytest.fixture(autouse=True, scope='module')
def create_test_database() -> Generator[None, None, None]:
    for url in DATABASE_URLS:
        database_url: DatabaseURL = DatabaseURL(url)
        if database_url.scheme in ['mysql', 'mysql+aiomysql', 'mysql+asyncmy']:
            url = str(database_url.replace(driver='pymysql'))
        elif database_url.scheme in ['postgresql+aiopg', 'sqlite+aiosqlite', 'postgresql+asyncpg']:
            url = str(database_url.replace(driver=None))
        engine: sqlalchemy.engine.Engine = sqlalchemy.create_engine(url)
        metadata.create_all(engine)
    yield
    for url in DATABASE_URLS:
        database_url: DatabaseURL = DatabaseURL(url)
        if database_url.scheme in ['mysql', 'mysql+aiomysql', 'mysql+asyncmy']:
            url = str(database_url.replace(driver='pymysql'))
        elif database_url.scheme in ['postgresql+aiopg', 'sqlite+aiosqlite', 'postgresql+asyncpg']:
            url = str(database_url.replace(driver=None))
        engine: sqlalchemy.engine.Engine = sqlalchemy.create_engine(url)
        metadata.drop_all(engine)

def get_app(database_url: str) -> Starlette:
    database: Database = Database(database_url, force_rollback=True)
    app: Starlette = Starlette()

    @app.on_event('startup')
    async def startup() -> None:
        await database.connect()

    @app.on_event('shutdown')
    async def shutdown() -> None:
        await database.disconnect()

    @app.route('/notes', methods=['GET'])
    async def list_notes(request: Request) -> JSONResponse:
        query: sqlalchemy.sql.Select = notes.select()
        results: List[Dict[str, Any]] = await database.fetch_all(query)
        content: List[Dict[str, Any]] = [{'text': result['text'], 'completed': result['completed']} for result in results]
        return JSONResponse(content)

    @app.route('/notes', methods=['POST'])
    async def add_note(request: Request) -> JSONResponse:
        data: Dict[str, Any] = await request.json()
        query: sqlalchemy.sql.Insert = notes.insert().values(text=data['text'], completed=data['completed'])
        await database.execute(query)
        return JSONResponse({'text': data['text'], 'completed': data['completed']})
    return app

@pytest.mark.parametrize('database_url', DATABASE_URLS)
def test_integration(database_url: str) -> None:
    app: Starlette = get_app(database_url)
    with TestClient(app) as client:
        response: Any = client.post('/notes', json={'text': 'example', 'completed': True})
        assert response.status_code == 200
        assert response.json() == {'text': 'example', 'completed': True}
        response: Any = client.get('/notes')
        assert response.status_code == 200
        assert response.json() == [{'text': 'example', 'completed': True}]
    with TestClient(app) as client:
        response: Any = client.get('/notes')
        assert response.status_code == 200
        assert response.json() == []
