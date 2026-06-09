from typing import Any

# === Third-party dependency: attr ===
# Used symbols: asdict

# === Internal dependency: chalice.app ===
class ScheduleExpression(object): ...
class S3EventConfig(BaseEventSourceConfig): ...
class SNSEventConfig(BaseEventSourceConfig): ...
class SQSEventConfig(BaseEventSourceConfig): ...
class ScheduledEventConfig(BaseEventSourceConfig): ...
class CloudWatchEventConfig(BaseEventSourceConfig): ...

# === Internal dependency: chalice.constants ===
LAMBDA_TRUST_POLICY: Any

# === Internal dependency: chalice.deploy.models ===
class Placeholder(enum.Enum):
    BUILD_STAGE: str
class RoleTraits(enum.Enum):
    VPC_NEEDED: str
class APIType(enum.Enum):
    WEBSOCKET: str
    HTTP: str
class TLSVersion(enum.Enum):
    ...
class Application(Model): ...
class DeploymentPackage(Model):
class IAMPolicy(Model):
class FileBasedIAMPolicy(IAMPolicy):
class AutoGenIAMPolicy(IAMPolicy):
class PreCreatedIAMRole(IAMRole): ...
class ManagedIAMRole(IAMRole, ManagedModel): ...
class LambdaLayer(ManagedModel):
class LambdaFunction(ManagedModel):
class CloudWatchEvent(CloudWatchEventBase):
class ScheduledEvent(CloudWatchEventBase):
class APIMapping(ManagedModel): ...
class DomainName(ManagedModel):
class RestAPI(ManagedModel): ...
class WebsocketAPI(ManagedModel): ...
class S3BucketNotification(FunctionEventSubscriber):
class SNSLambdaSubscription(FunctionEventSubscriber):
class SQSEventSource(FunctionEventSubscriber):