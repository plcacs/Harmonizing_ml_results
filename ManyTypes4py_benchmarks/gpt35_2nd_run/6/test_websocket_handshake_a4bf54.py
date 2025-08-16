def gen_ws_headers(protocols: str = '', compress: int = 0, extension_text: str = '', server_notakeover: bool = False, client_notakeover: bool = False) -> Tuple[List[Tuple[str, str]], str]:

async def test_handshake_protocol_unsupported(caplog: Any):

def test_handshake_compress_server_notakeover():

def test_handshake_compress_client_notakeover():

def test_handshake_compress_wbits():

def test_handshake_compress_wbits_error():

def test_handshake_compress_bad_ext():

def test_handshake_compress_multi_ext_bad():

def test_handshake_compress_multi_ext_wbits():

def test_handshake_no_transfer_encoding():
