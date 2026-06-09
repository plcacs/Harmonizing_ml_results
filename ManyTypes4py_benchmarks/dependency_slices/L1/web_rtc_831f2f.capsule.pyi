from typing import Any

# === Third-party dependency: aiortc ===
# Used symbols: InvalidStateError, RTCDataChannel, RTCPeerConnection, RTCSessionDescription

# === Third-party dependency: aiortc.sdp ===
def candidate_from_sdp(sdp: str) -> RTCIceCandidate: ...
def candidate_to_sdp(candidate: RTCIceCandidate) -> str: ...

# === Third-party dependency: gevent ===
# Used symbols: Greenlet, killall, spawn

# === Third-party dependency: gevent.lock ===
# Used symbols: Semaphore

# === Internal dependency: raiden.network.transport.matrix.client ===
class ReceivedRaidenMessage(_ReceivedMessageBase): ...

# === Internal dependency: raiden.network.transport.matrix.rtc.aiogevent ===
def yield_future(future, loop=...): ...
def wrap_greenlet(gt, loop=...): ...

# === Internal dependency: raiden.network.transport.matrix.utils ===
def validate_and_parse_message(data, peer_address): ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address): ...

# === Internal dependency: raiden.utils.runnable ===
class Runnable:
    ...

# === Internal dependency: raiden.utils.typing ===
from typing import Any
from typing import Dict
from eth_typing import Address

# === Third-party dependency: structlog ===
# Used symbols: get_logger