from typing import Any, Dict, List, Optional, cast
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

    The OAuth2 specification dictates that for a password flow the data should be
    collected using form data (instead of JSON) and that it should have the specific
    fields `username` and `password`.

    All the initialization parameters are extracted from the request.

    Read more about it in the
    [FastAPI docs for Simple OAuth2 with Password and Bearer](https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/).

    ## Example

    ```python
    from typing import Annotated

    from fastapi import Depends, FastAPI
    from fastapi.security import OAuth2PasswordRequestForm

    app = FastAPI()


    @app.post("/login")
    def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
        data = {}
        data["scopes"] = []
        for scope in form_data.scopes:
            data["scopes"].append(scope)
        if form_data.client_id:
            data["client_id"] = form_data.client_id
        if form_data.client_secret:
            data["client_secret"] = form_data.client_secret
        return data
    ```

    Note that for OAuth2 the scope `items:read` is a single scope in an opaque string.
    You could have custom internal logic to separate it by colon characters (`:`) or
    similar, and get the two parts `items` and `read`. Many applications do that to
    group and organize permissions, you could do it as well in your application, just
    know that that it is application specific, it's not part of the specification.
    """

    def __init__(
        self,
        *,
        grant_type: Optional[str] = None,
        username: str,
        password: str,
        scope: str = '',
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
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
    for an OAuth2 password flow.

    The OAuth2 specification dictates that for a password flow the data should be
    collected using form data (instead of JSON) and that it should have the specific
    fields `username` and `password`.

    All the initialization parameters are extracted from the request.

    The only difference between `OAuth2PasswordRequestFormStrict` and
    `OAuth2PasswordRequestForm` is that `OAuth2PasswordRequestFormStrict` requires the
    client to send the form field `grant_type` with the value `"password"`, which
    is required in the OAuth2 specification (it seems that for no particular reason),
    while for `OAuth2PasswordRequestForm` `grant_type` is optional.

    Read more about it in the
    [FastAPI docs for Simple OAuth2 with Password and Bearer](https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/).

    ## Example

    ```python
    from typing import Annotated

    from fastapi import Depends, FastAPI
    from fastapi.security import OAuth2PasswordRequestForm

    app = FastAPI()


    @app.post("/login")
    def login(form_data: Annotated[OAuth2PasswordRequestFormStrict, Depends()]):
        data = {}
        data["scopes"] = []
        for scope in form_data.scopes:
            data["scopes"].append(scope)
        if form_data.client_id:
            data["client_id"] = form_data.client_id
        if form_data.client_secret:
            data["client_secret"] = form_data.client_secret
        return data
    ```

    Note that for OAuth2 the scope `items:read` is a single scope in an opaque string.
    You could have custom internal logic to separate it by colon characters (`:`) or
    similar, and get the two parts `items` and `read`. Many applications do that to
    group and organize permissions, you could do it as well in your application, just
    know that that it is application specific, it's not part of the specification.


    grant_type: the OAuth2 spec says it is required and MUST be the fixed string "password".
        This dependency is strict about it. If you want to be permissive, use instead the
        OAuth2PasswordRequestForm dependency class.
    username: username string. The OAuth2 spec requires the exact field name "username".
    password: password string. The OAuth2 spec requires the exact field name "password".
    scope: Optional string. Several scopes (each one a string) separated by spaces. E.g.
        "items:read items:write users:read profile openid"
    client_id: optional string. OAuth2 recommends sending the client_id and client_secret (if any)
        using HTTP Basic auth, as: client_id:client_secret
    client_secret: optional string. OAuth2 recommends sending the client_id and client_secret (if any)
        using HTTP Basic auth, as: client_id:client_secret
    """

    def __init__(
        self,
        grant_type: str,
        username: str,
        password: str,
        scope: str = '',
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        super().__init__(
            grant_type=grant_type,
            username=username,
            password=password,
            scope=scope,
            client_id=client_id,
            client_secret=client_secret,
        )

class OAuth2(SecurityBase):
    """
    This is the base class for OAuth2 authentication, an instance of it would be used
    as a dependency. All other OAuth2 classes inherit from it and customize it for
    each OAuth2 flow.

    You normally would not create a new class inheriting from it but use one of the
    existing subclasses, and maybe compose them if you want to support multiple flows.

    Read more about it in the
    [FastAPI docs for Security](https://fastapi.tiangolo.com/tutorial/security/).
    """

    def __init__(
        self,
        *,
        flows: OAuthFlowsModel = OAuthFlowsModel(),
        scheme_name: Optional[str] = None,
        description: Optional[str] = None,
        auto_error: bool = True,
    ) -> None:
        self.model: OAuth2Model = OAuth2Model(
            flows=cast(OAuthFlowsModel, flows), description=description
        )
        self.scheme_name: str = scheme_name or self.__class__.__name__
        self.auto_error: bool = auto_error

    async def __call__(self, request: Request) -> Optional[str]:
        authorization = request.headers.get('Authorization')
        if not authorization:
            if self.auto_error:
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail='Not authenticated')
            else:
                return None
        return authorization

class OAuth2PasswordBearer(OAuth2):
    """
    OAuth2 flow for authentication using a bearer token obtained with a password.
    An instance of it would be used as a dependency.

    Read more about it in the
    [FastAPI docs for Simple OAuth2 with Password and Bearer](https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/).
    """

    def __init__(
        self,
        tokenUrl: str,
        scheme_name: Optional[str] = None,
        scopes: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        auto_error: bool = True,
    ) -> None:
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password=cast(Any, {'tokenUrl': tokenUrl, 'scopes': scopes}))
        super().__init__(flows=flows, scheme_name=scheme_name, description=description, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization = request.headers.get('Authorization')
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != 'bearer':
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail='Not authenticated',
                    headers={'WWW-Authenticate': 'Bearer'},
                )
            else:
                return None
        return param

class OAuth2AuthorizationCodeBearer(OAuth2):
    """
    OAuth2 flow for authentication using a bearer token obtained with an OAuth2 code
    flow. An instance of it would be used as a dependency.
    """

    def __init__(
        self,
        authorizationUrl: str,
        tokenUrl: str,
        refreshUrl: Optional[str] = None,
        scheme_name: Optional[str] = None,
        scopes: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        auto_error: bool = True,
    ) -> None:
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(
            authorizationCode=cast(
                Any,
                {'authorizationUrl': authorizationUrl, 'tokenUrl': tokenUrl, 'refreshUrl': refreshUrl, 'scopes': scopes},
            )
        )
        super().__init__(flows=flows, scheme_name=scheme_name, description=description, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization = request.headers.get('Authorization')
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != 'bearer':
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail='Not authenticated',
                    headers={'WWW-Authenticate': 'Bearer'},
                )
            else:
                return None
        return param

class SecurityScopes:
    """
    This is a special class that you can define in a parameter in a dependency to
    obtain the OAuth2 scopes required by all the dependencies in the same chain.

    This way, multiple dependencies can have different scopes, even when used in the
    same *path operation*. And with this, you can access all the scopes required in
    all those dependencies in a single place.

    Read more about it in the
    [FastAPI docs for OAuth2 scopes](https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/).
    """

    def __init__(self, scopes: Optional[List[str]] = None) -> None:
        self.scopes: List[str] = scopes or []
        self.scope_str: str = ' '.join(self.scopes)