    def set_cookie(self, cookie: Optional[str]) -> None:
    def formatfundajson(fundajson: dict) -> dict:
    def formatfundbjson(fundbjson: dict) -> dict:
    def formatetfindexjson(fundbjson: dict) -> dict:
    def formatjisilujson(data: dict) -> dict:
    def percentage2float(per: str) -> float:
    def funda(self, fields: Optional[list] = None, min_volume: int = 0, min_discount: int = 0, ignore_nodown: bool = False, forever: bool = False) -> dict:
    def fundm(self) -> dict:
    def fundb(self, fields: Optional[list] = None, min_volume: int = 0, min_discount: int = 0, forever: bool = False) -> dict:
    def fundarb(self, jsl_username: str, jsl_password: str, avolume: int = 100, bvolume: int = 100, ptype: str = 'price') -> dict:
    def etfindex(self, index_id: str = '', min_volume: int = 0, max_discount: Optional[str] = None, min_discount: Optional[str] = None) -> dict:
    def qdii(self, min_volume: int = 0) -> dict:
    def cb(self, min_volume: int = 0, cookie: Optional[str] = None) -> dict:
