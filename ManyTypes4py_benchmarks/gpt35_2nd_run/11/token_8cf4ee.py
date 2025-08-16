    def __init__(self, iss: str, typ: str, sub: str, aud: str, exp: int, nbf: int, iat: int, jti: str = None, **kwargs: Any) -> None:

    @classmethod
    def parse(cls, token: str, key: str = None, verify: bool = True, algorithm: str = 'HS256') -> 'Jwt':

    @property
    def serialize(self) -> Dict[str, Any]:

    def tokenize(self, algorithm: str = 'HS256') -> str:
