from typing import NamedTuple, List, Optional
from raiden.transfer.state import NettingChannelState
from raiden.transfer.state_change import Block
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed
from raiden.transfer.events import ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state_change import ReceiveUnlock
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, ReceiveUnlock
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, ReceiveUnlock
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, ReceiveUnlock
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block
from raiden.transfer.events import SendSecretRequest, SendSecretReveal, EventUnlockClaimFailed, ContractSendSecretReveal, SendProcessed, ContractReceiveSecretReveal
from raiden.transfer.state import NettingChannelState
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden