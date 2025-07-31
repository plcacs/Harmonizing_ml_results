from typing import Any, Dict, List, Optional, Union, cast
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import OAuth2 as OAuth2Model
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.param_functions import Form
from fastapi.security.base import SecurityBase
from fastapi.security.utils import get_authorization_scheme_param
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from typing_extensions import Annotated, Doc

class OAuth2PasswordRequestForm:
    """
    This is a dependency class to collect the `username` and `password` as form data
    for an OAuth2 password flow.
    """
    def __init__(
        self,
        *,
        grant_type: Optional[str] = None,
        username: str,
        password: str,
        scope: str = '',
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ) -> None:
        self.grant_type: Optional[str] = grant_type
        self.username: str = username
        self.password: str = password
        self.scopes: List[str] = scope.split()
        self.client_id: Optional[str] = client_id
        self.client_secret: Optional[str] = client_secret

class OAuth2PasswordRequestFormStrict(OAuth2PasswordRequestForm):
    """
    This is a dependency class to collect the `username` and `password` as form data
    for an OAuth2 password flow. It requires the form field `grant_type` with the value "password".
    """
    def __init__(
        self,
        grant_type: str,
        username: str,
        password: str,
        scope: str = '',
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ) -> None:
        super().__init__(
            grant_type=grant_type,
            username=username,
            password=password,
            scope=scope,
            client_id=client_id,
            client_secret=client_secret
        )

class OAuth2(SecurityBase):
    """
    This is the base class for OAuth2 authentication used as a dependency.
    """
    def __init__(
        self,
        *,
        flows: OAuthFlowsModel = OAuthFlowsModel(),
        scheme_name: Optional[str] = None,
        description: Optional[str] = None,
        auto_error: bool = True
    ) -> None:
        self.model: OAuth2Model = OAuth2Model(flows=cast(OAuthFlowsModel, flows), description=description)
        self.scheme_name: str = scheme_name or self.__class__.__name__
        self.auto_error: bool = auto_error

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: Optional[str] = request.headers.get('Authorization')
        if not authorization:
            if self.auto_error:
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail='Not authenticated')
            else:
                return None
        return authorization

class OAuth2PasswordBearer(OAuth2):
    """
    OAuth2 flow for authentication using a bearer token obtained with a password.
    """
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: Optional[str] = None,
        scopes: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        auto_error: bool = True
    ) -> None:
        if not scopes:
            scopes = {}
        flows: OAuthFlowsModel = OAuthFlowsModel(password=cast(Any, {'tokenUrl': tokenUrl, 'scopes': scopes}))
        super().__init__(flows=flows, scheme_name=scheme_name, description=description, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: Optional[str] = request.headers.get('Authorization')
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != 'bearer':
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail='Not authenticated',
                    headers={'WWW-Authenticate': 'Bearer'}
                )
            else:
                return None
        return param

class OAuth2AuthorizationCodeBearer(OAuth2):
    """
    OAuth2 flow for authentication using a bearer token obtained with an OAuth2 code flow.
    """
    def __init__(
        self,
        authorizationUrl: str,
        tokenUrl: str,
        refreshUrl: Optional[str] = None,
        scheme_name: Optional[str] = None,
        scopes: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        auto_error: bool = True
    ) -> None:
        if not scopes:
            scopes = {}
        flows: OAuthFlowsModel = OAuthFlowsModel(
            authorizationCode=cast(Any, {
                'authorizationUrl': authorizationUrl,
                'tokenUrl': tokenUrl,
                'refreshUrl': refreshUrl,
                'scopes': scopes
            })
        )
        super().__init__(flows=flows, scheme_name=scheme_name, description=description, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization: Optional[str] = request.headers.get('Authorization')
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != 'bearer':
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail='Not authenticated',
                    headers={'WWW-Authenticate': 'Bearer'}
                )
            else:
                return None
        return param

class SecurityScopes:
    """
    A class to obtain the OAuth2 scopes required by dependencies in the same chain.
    """
    def __init__(self, scopes: Optional[List[str]] = None) -> None:
        self.scopes: List[str] = scopes or []
        self.scope_str: str = ' '.join(self.scopes)