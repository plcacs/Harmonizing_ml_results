import warnings
import pytest
from fastapi import APIRouter, Depends, FastAPI, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from typing import Any, Dict

class ResponseLevel0(JSONResponse):
    media_type: str = 'application/x-level-0'

class ResponseLevel1(JSONResponse):
    media_type: str = 'application/x-level-1'

class ResponseLevel2(JSONResponse):
    media_type: str = 'application/x-level-2'

class ResponseLevel3(JSONResponse):
    media_type: str = 'application/x-level-3'

class ResponseLevel4(JSONResponse):
    media_type: str = 'application/x-level-4'

class ResponseLevel5(JSONResponse):
    media_type: str = 'application/x-level-5'

async def dep0(response: Response) -> None:
    response.headers['x-level0'] = 'True'

async def dep1(response: Response) -> None:
    response.headers['x-level1'] = 'True'

async def dep2(response: Response) -> None:
    response.headers['x-level2'] = 'True'

async def dep3(response: Response) -> None:
    response.headers['x-level3'] = 'True'

async def dep4(response: Response) -> None:
    response.headers['x-level4'] = 'True'

async def dep5(response: Response) -> None:
    response.headers['x-level5'] = 'True'

callback_router0: APIRouter = APIRouter()

@callback_router0.get('/')
async def callback0(level0: str) -> None:
    pass

callback_router1: APIRouter = APIRouter()

@callback_router1.get('/')
async def callback1(level1: str) -> None:
    pass

callback_router2: APIRouter = APIRouter()

@callback_router2.get('/')
async def callback2(level2: str) -> None:
    pass

callback_router3: APIRouter = APIRouter()

@callback_router3.get('/')
async def callback3(level3: str) -> None:
    pass

callback_router4: APIRouter = APIRouter()

@callback_router4.get('/')
async def callback4(level4: str) -> None:
    pass

callback_router5: APIRouter = APIRouter()

@callback_router5.get('/')
async def callback5(level5: str) -> None:
    pass

app: FastAPI = FastAPI(
    dependencies=[Depends(dep0)],
    responses={400: {'description': 'Client error level 0'}, 500: {'description': 'Server error level 0'}},
    default_response_class=ResponseLevel0,
    callbacks=callback_router0.routes
)

router2_override: APIRouter = APIRouter(
    prefix='/level2',
    tags=['level2a', 'level2b'],
    dependencies=[Depends(dep2)],
    responses={402: {'description': 'Client error level 2'}, 502: {'description': 'Server error level 2'}},
    default_response_class=ResponseLevel2,
    callbacks=callback_router2.routes,
    deprecated=True
)

router2_default: APIRouter = APIRouter()

router4_override: APIRouter = APIRouter(
    prefix='/level4',
    tags=['level4a', 'level4b'],
    dependencies=[Depends(dep4)],
    responses={404: {'description': 'Client error level 4'}, 504: {'description': 'Server error level 4'}},
    default_response_class=ResponseLevel4,
    callbacks=callback_router4.routes,
    deprecated=True
)

router4_default: APIRouter = APIRouter()

@app.get(
    '/override1',
    tags=['path1a', 'path1b'],
    responses={401: {'description': 'Client error level 1'}, 501: {'description': 'Server error level 1'}},
    deprecated=True,
    callbacks=callback_router1.routes,
    dependencies=[Depends(dep1)],
    response_class=ResponseLevel1,
)
async def path1_override(level1: str) -> str:
    return level1

@app.get('/default1')
async def path1_default(level1: str) -> str:
    return level1

@router2_override.get(
    '/override3',
    tags=['path3a', 'path3b'],
    responses={403: {'description': 'Client error level 3'}, 503: {'description': 'Server error level 3'}},
    deprecated=True,
    callbacks=callback_router3.routes,
    dependencies=[Depends(dep3)],
    response_class=ResponseLevel3,
)
async def path3_override_router2_override(level3: str) -> str:
    return level3

@router2_override.get('/default3')
async def path3_default_router2_override(level3: str) -> str:
    return level3

@router2_default.get(
    '/override3',
    tags=['path3a', 'path3b'],
    responses={403: {'description': 'Client error level 3'}, 503: {'description': 'Server error level 3'}},
    deprecated=True,
    callbacks=callback_router3.routes,
    dependencies=[Depends(dep3)],
    response_class=ResponseLevel3,
)
async def path3_override_router2_default(level3: str) -> str:
    return level3

@router2_default.get('/default3')
async def path3_default_router2_default(level3: str) -> str:
    return level3

@router4_override.get(
    '/override5',
    tags=['path5a', 'path5b'],
    responses={405: {'description': 'Client error level 5'}, 505: {'description': 'Server error level 5'}},
    deprecated=True,
    callbacks=callback_router5.routes,
    dependencies=[Depends(dep5)],
    response_class=ResponseLevel5,
)
async def path5_override_router4_override(level5: str) -> str:
    return level5

@router4_override.get('/default5')
async def path5_default_router4_override(level5: str) -> str:
    return level5

@router4_default.get(
    '/override5',
    tags=['path5a', 'path5b'],
    responses={405: {'description': 'Client error level 5'}, 505: {'description': 'Server error level 5'}},
    deprecated=True,
    callbacks=callback_router5.routes,
    dependencies=[Depends(dep5)],
    response_class=ResponseLevel5,
)
async def path5_override_router4_default(level5: str) -> str:
    return level5

@router4_default.get('/default5')
async def path5_default_router4_default(level5: str) -> str:
    return level5

router2_override.include_router(
    router4_override,
    prefix='/level3',
    tags=['level3a', 'level3b'],
    dependencies=[Depends(dep3)],
    responses={403: {'description': 'Client error level 3'}, 503: {'description': 'Server error level 3'}},
    default_response_class=ResponseLevel3,
    callbacks=callback_router3.routes,
)
router2_override.include_router(
    router4_default,
    prefix='/level3',
    tags=['level3a', 'level3b'],
    dependencies=[Depends(dep3)],
    responses={403: {'description': 'Client error level 3'}, 503: {'description': 'Server error level 3'}},
    default_response_class=ResponseLevel3,
    callbacks=callback_router3.routes,
)
router2_override.include_router(router4_override)
router2_override.include_router(router4_default)
router2_default.include_router(
    router4_override,
    prefix='/level3',
    tags=['level3a', 'level3b'],
    dependencies=[Depends(dep3)],
    responses={403: {'description': 'Client error level 3'}, 503: {'description': 'Server error level 3'}},
    default_response_class=ResponseLevel3,
    callbacks=callback_router3.routes,
)
router2_default.include_router(
    router4_default,
    prefix='/level3',
    tags=['level3a', 'level3b'],
    dependencies=[Depends(dep3)],
    responses={403: {'description': 'Client error level 3'}, 503: {'description': 'Server error level 3'}},
    default_response_class=ResponseLevel3,
    callbacks=callback_router3.routes,
)
router2_default.include_router(router4_override)
router2_default.include_router(router4_default)
app.include_router(
    router2_override,
    prefix='/level1',
    tags=['level1a', 'level1b'],
    dependencies=[Depends(dep1)],
    responses={401: {'description': 'Client error level 1'}, 501: {'description': 'Server error level 1'}},
    default_response_class=ResponseLevel1,
    callbacks=callback_router1.routes,
)
app.include_router(
    router2_default,
    prefix='/level1',
    tags=['level1a', 'level1b'],
    dependencies=[Depends(dep1)],
    responses={401: {'description': 'Client error level 1'}, 501: {'description': 'Server error level 1'}},
    default_response_class=ResponseLevel1,
    callbacks=callback_router1.routes,
)
app.include_router(router2_override)
app.include_router(router2_default)
client: TestClient = TestClient(app)

def test_level1_override() -> None:
    response = client.get('/override1?level1=foo')
    assert response.json() == 'foo'
    assert response.headers['content-type'] == 'application/x-level-1'
    assert 'x-level0' in response.headers
    assert 'x-level1' in response.headers
    assert 'x-level2' not in response.headers
    assert 'x-level3' not in response.headers
    assert 'x-level4' not in response.headers
    assert 'x-level5' not in response.headers

def test_level1_default() -> None:
    response = client.get('/default1?level1=foo')
    assert response.json() == 'foo'
    assert response.headers['content-type'] == 'application/x-level-0'
    assert 'x-level0' in response.headers
    assert 'x-level1' not in response.headers
    assert 'x-level2' not in response.headers
    assert 'x-level3' not in response.headers
    assert 'x-level4' not in response.headers
    assert 'x-level5' not in response.headers

@pytest.mark.parametrize('override1', [True, False])
@pytest.mark.parametrize('override2', [True, False])
@pytest.mark.parametrize('override3', [True, False])
def test_paths_level3(override1: bool, override2: bool, override3: bool) -> None:
    url: str = ''
    content_type_level: str = '0'
    if override1:
        url += '/level1'
        content_type_level = '1'
    if override2:
        url += '/level2'
        content_type_level = '2'
    if override3:
        url += '/override3'
        content_type_level = '3'
    else:
        url += '/default3'
    url += '?level3=foo'
    response = client.get(url)
    assert response.json() == 'foo'
    assert response.headers['content-type'] == f'application/x-level-{content_type_level}'
    assert 'x-level0' in response.headers
    if override1:
        assert 'x-level1' in response.headers
    if override2:
        assert 'x-level2' in response.headers
    if override3:
        assert 'x-level3' in response.headers

@pytest.mark.parametrize('override1', [True, False])
@pytest.mark.parametrize('override2', [True, False])
@pytest.mark.parametrize('override3', [True, False])
@pytest.mark.parametrize('override4', [True, False])
@pytest.mark.parametrize('override5', [True, False])
def test_paths_level5(override1: bool, override2: bool, override3: bool, override4: bool, override5: bool) -> None:
    url: str = ''
    content_type_level: str = '0'
    if override1:
        url += '/level1'
        content_type_level = '1'
    if override2:
        url += '/level2'
        content_type_level = '2'
    if override3:
        url += '/level3'
        content_type_level = '3'
    if override4:
        url += '/level4'
        content_type_level = '4'
    if override5:
        url += '/override5'
        content_type_level = '5'
    else:
        url += '/default5'
    url += '?level5=foo'
    response = client.get(url)
    assert response.json() == 'foo'
    assert response.headers['content-type'] == f'application/x-level-{content_type_level}'
    assert 'x-level0' in response.headers
    if override1:
        assert 'x-level1' in response.headers
    if override2:
        assert 'x-level2' in response.headers
    if override3:
        assert 'x-level3' in response.headers
    if override4:
        assert 'x-level4' in response.headers
    if override5:
        assert 'x-level5' in response.headers

def test_openapi() -> None:
    client_local: TestClient = TestClient(app)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        response = client_local.get('/openapi.json')
        assert issubclass(w[-1].category, UserWarning)
        assert 'Duplicate Operation ID' in str(w[-1].message)
    # The full expected openapi specification is omitted for brevity.
    openapi: Dict[str, Any] = response.json()
    assert isinstance(openapi, dict)