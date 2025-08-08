from __future__ import annotations
import abc
import re
import weakref
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Type, TypeVar, Union
from uuid import UUID, uuid4
from pydantic import Field, PrivateAttr, field_validator, model_validator
from typing_extensions import Self, TypeAlias
from prefect.logging import get_logger
from prefect.server.events.actions import ServerActionTypes
from prefect.server.events.schemas.events import ReceivedEvent, RelatedResource, Resource, ResourceSpecification, matches
from prefect.server.schemas.actions import ActionBaseModel
from prefect.server.utilities.schemas import ORMBaseModel, PrefectBaseModel
from prefect.types import DateTime
from prefect.utilities.collections import AutoEnum
if TYPE_CHECKING:
    import logging
logger = get_logger(__name__)

class Posture(AutoEnum):
    Reactive: str = 'Reactive'
    Proactive: str = 'Proactive'
    Metric: str = 'Metric'

class TriggerState(AutoEnum):
    Triggered: str = 'Triggered'
    Resolved: str = 'Resolved'

class Trigger(PrefectBaseModel, abc.ABC):
    id: UUID = Field(default_factory=uuid4, description='The unique ID of this trigger')
    _automation: Any = PrivateAttr(None)
    _parent: Any = PrivateAttr(None)

    @property
    def automation(self) -> Any:
        assert self._automation is not None, 'Trigger._automation has not been set'
        value = self._automation()
        assert value is not None, 'Trigger._automation has been garbage collected'
        return value

    @property
    def parent(self) -> Any:
        assert self._parent is not None, 'Trigger._parent has not been set'
        value = self._parent()
        assert value is not None, 'Trigger._parent has been garbage collected'
        return value

    def _set_parent(self, value: Any) -> None:
        if isinstance(value, Automation):
            self._automation = weakref.ref(value)
            self._parent = self._automation
        elif isinstance(value, Trigger):
            self._parent = weakref.ref(value)
            self._automation = value._automation
        else:
            raise ValueError('parent must be an Automation or a Trigger')

    def reset_ids(self) -> None:
        """Resets the ID of this trigger and all of its children"""
        self.id = uuid4()
        for trigger in self.all_triggers():
            trigger.id = uuid4()

    def all_triggers(self) -> List[Trigger]:
        """Returns all triggers within this trigger"""
        return [self]

    @abc.abstractmethod
    def create_automation_state_change_event(self, firing: Firing, trigger_state: TriggerState) -> None:
        ...

class CompositeTrigger(Trigger, abc.ABC):
    def create_automation_state_change_event(self, firing: Firing, trigger_state: TriggerState) -> ReceivedEvent:
        ...

    def _set_parent(self, value: Any) -> None:
        ...

    def all_triggers(self) -> List[Trigger]:
        ...

    @property
    def child_trigger_ids(self) -> List[UUID]:
        ...

    @property
    def num_expected_firings(self) -> int:
        ...

    @abc.abstractmethod
    def ready_to_fire(self, firings: List[Firing]) -> bool:
        ...

class CompoundTrigger(CompositeTrigger):
    type: str = 'compound'

    @property
    def num_expected_firings(self) -> int:
        ...

    def ready_to_fire(self, firings: List[Firing]) -> bool:
        ...

    @model_validator(mode='after')
    def validate_require(self) -> CompoundTrigger:
        ...

class SequenceTrigger(CompositeTrigger):
    type: str = 'sequence'

    @property
    def expected_firing_order(self) -> List[UUID]:
        ...

    def ready_to_fire(self, firings: List[Firing]) -> bool:
        ...

class ResourceTrigger(Trigger, abc.ABC):
    match: ResourceSpecification = Field(default_factory=lambda: ResourceSpecification.model_validate({}), description='Labels for resources which this trigger will match.')
    match_related: ResourceSpecification = Field(default_factory=lambda: ResourceSpecification.model_validate({}), description='Labels for related resources which this trigger will match.')

    def covers_resources(self, resource: Resource, related: List[RelatedResource]) -> bool:
        ...

class EventTrigger(ResourceTrigger):
    type: str = 'event'
    after: Set[str] = Field(default_factory=set, description='The event(s) which must first been seen to fire this trigger.  If empty, then fire this trigger immediately.  Events may include trailing wildcards, like `prefect.flow-run.*`')
    expect: Set[str] = Field(default_factory=set, description='The event(s) this trigger is expecting to see.  If empty, this trigger will match any event.  Events may include trailing wildcards, like `prefect.flow-run.*`')
    for_each: Set[str] = Field(default_factory=set, description='Evaluate the trigger separately for each distinct value of these labels on the resource.')
    posture: Posture = Field(..., description='The posture of this trigger, either Reactive or Proactive.')
    threshold: int = Field(1, description='The number of events required for this trigger to fire (for Reactive triggers), or the number of events expected (for Proactive triggers)')
    within: timedelta = Field(timedelta(seconds=0), ge=timedelta(seconds=0), description='The time period over which the events must occur.')

    @model_validator(mode='before')
    @classmethod
    def enforce_minimum_within_for_proactive_triggers(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def covers(self, event: ReceivedEvent) -> bool:
        ...

    @property
    def immediate(self) -> bool:
        ...

    @property
    def event_pattern(self) -> re.Pattern:
        ...

    def starts_after(self, event: ReceivedEvent) -> bool:
        ...

    def expects(self, event: ReceivedEvent) -> bool:
        ...

    def bucketing_key(self, event: ReceivedEvent) -> Tuple[str, ...]:
        ...

    def meets_threshold(self, event_count: int) -> bool:
        ...

    def create_automation_state_change_event(self, firing: Firing, trigger_state: TriggerState) -> ReceivedEvent:
        ...

ServerTriggerTypes = Union[EventTrigger, CompoundTrigger, SequenceTrigger]
T = TypeVar('T', bound=Trigger)

class AutomationCore(PrefectBaseModel, extra='ignore'):
    name: str = Field(default=..., description='The name of this automation')
    description: str = Field(default='', description='A longer description of this automation')
    enabled: bool = Field(default=True, description='Whether this automation will be evaluated')
    trigger: Trigger = Field(default=..., description='The criteria for which events this Automation covers and how it will respond to the presence or absence of those events')
    actions: Any = Field(default=..., description='The actions to perform when this Automation triggers')
    actions_on_trigger: List[Any] = Field(default_factory=list, description='The actions to perform when an Automation goes into a triggered state')
    actions_on_resolve: List[Any] = Field(default_factory=list, description='The actions to perform when an Automation goes into a resolving state')

    def triggers(self) -> List[Trigger]:
        ...

    def triggers_of_type(self, trigger_type: Type[Trigger]) -> List[Trigger]:
        ...

    def trigger_by_id(self, trigger_id: UUID) -> Optional[Trigger]:
        ...

    @model_validator(mode='after')
    def prevent_run_deployment_loops(self) -> AutomationCore:
        ...

class Automation(ORMBaseModel, AutomationCore, extra='ignore'):
    def __init__(self, *args, **kwargs):
        ...

    @classmethod
    def model_validate(cls, obj: Dict[str, Any], *, strict: Optional[bool] = None, from_attributes: Optional[bool] = None, context: Optional[Dict[str, Any]] = None) -> Automation:
        ...

class AutomationCreate(AutomationCore, ActionBaseModel, extra='forbid'):
    owner_resource: Optional[Any] = Field(default=None, description='The resource to which this automation belongs')

class AutomationUpdate(AutomationCore, ActionBaseModel, extra='forbid'):
    pass

class AutomationPartialUpdate(ActionBaseModel, extra='forbid'):
    enabled: bool = Field(True, description='Whether this automation will be evaluated')

class AutomationSort(AutoEnum):
    CREATED_DESC: str = 'CREATED_DESC'
    UPDATED_DESC: str = 'UPDATED_DESC'
    NAME_ASC: str = 'NAME_ASC'
    NAME_DESC: str = 'NAME_DESC'

class Firing(PrefectBaseModel):
    id: UUID = Field(default_factory=uuid4)
    trigger: Trigger = Field(default=..., description='The trigger that is firing')
    trigger_states: List[TriggerState] = Field(default=..., description='The state changes represented by this Firing')
    triggered: datetime = Field(default=..., description='The time at which this trigger fired')
    triggering_labels: Dict[str, str] = Field(default_factory=dict, description='The labels associated with this Firing')
    triggering_firings: List[Firing] = Field(default_factory=list, description='The firings of the triggers that caused this trigger to fire')
    triggering_event: Optional[ReceivedEvent] = Field(default=None, description='The most recent event associated with this Firing')
    triggering_value: Optional[Any] = Field(default=None, description='A value associated with this firing of a trigger')

    @field_validator('trigger_states')
    @classmethod
    def validate_trigger_states(cls, value: List[TriggerState]) -> List[TriggerState]:
        ...

    def all_firings(self) -> List[Firing]:
        ...

    def all_events(self) -> List[ReceivedEvent]:
        ...

class TriggeredAction(PrefectBaseModel):
    automation: Automation = Field(..., description='The Automation that caused this action')
    id: UUID = Field(default_factory=uuid4, description='A unique key representing a single triggering of an action')
    firing: Optional[Firing] = Field(None, description='The Firing that prompted this action')
    triggered: datetime = Field(..., description='When this action was triggered')
    triggering_labels: Dict[str, str] = Field(..., description="The subset of labels of the Event that triggered this action")
    triggering_event: ReceivedEvent = Field(..., description='The last Event to trigger this automation')
    action: Any = Field(..., description='The action to perform')
    action_index: int = Field(0, description='The index of the action within the automation')

    def idempotency_key(self) -> str:
        ...

    def all_firings(self) -> List[Firing]:
        ...

    def all_events(self) -> List[ReceivedEvent]:
        ...
