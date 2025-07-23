import time
from typing import Dict, List, Optional, Union
from googleapiclient.discovery import Resource
from pydantic import BaseModel, Field

class SecretKeySelector(BaseModel):
    """
    SecretKeySelector is a data model for specifying a GCP secret to inject
    into a Cloud Run V2 Job as an environment variable.
    Follows Cloud Run V2 rest API, docs:
    https://cloud.google.com/run/docs/reference/rest/v2/Container#SecretKeySelector
    """

class JobV2(BaseModel):
    """
    JobV2 is a data model for a job that will be run on Cloud Run with the V2 API.
    """
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    deleteTime: Optional[str] = Field(None)
    expireTime: Optional[str] = Field(None)
    creator: Optional[str] = Field(None)
    lastModifier: Optional[str] = Field(None)
    client: Optional[str] = Field(None)
    clientVersion: Optional[str] = Field(None)
    binaryAuthorization: Dict[str, Union[str, Dict]] = Field(default_factory=dict)
    template: Dict[str, Union[str, Dict]] = Field(default_factory=dict)
    observedGeneration: Optional[int] = Field(None)
    terminalCondition: Dict[str, Union[str, Dict]] = Field(default_factory=dict)
    conditions: List[Dict[str, Union[str, Dict]]] = Field(default_factory=list)
    latestCreatedExecution: Dict[str, Union[str, Dict]] = Field(default_factory=dict)
    reconciling: bool = Field(False)
    satisfiesPzs: bool = Field(False)
    etag: Optional[str] = Field(None)

    def is_ready(self) -> bool:
        """
        Check if the job is ready to run.

        Returns:
            Whether the job is ready to run.
        """
        ready_condition = self.get_ready_condition()
        if self._is_missing_container(ready_condition=ready_condition):
            raise Exception(f"{ready_condition.get('message')}")
        return ready_condition.get('state') == 'CONDITION_SUCCEEDED'

    def get_ready_condition(self) -> Dict[str, Union[str, Dict]]:
        """
        Get the ready condition for the job.

        Returns:
            The ready condition for the job.
        """
        if self.terminalCondition.get('type') == 'Ready':
            return self.terminalCondition
        return {}

    @classmethod
    def get(cls, cr_client: Resource, project: str, location: str, job_name: str) -> 'JobV2':
        """
        Get a job from Cloud Run with the V2 API.

        Args:
            cr_client: The base client needed for interacting with GCP
                Cloud Run V2 API.
            project: The GCP project ID.
            location: The GCP region.
            job_name: The name of the job to get.
        """
        request = cr_client.jobs().get(name=f'projects/{project}/locations/{location}/jobs/{job_name}')
        response = request.execute()
        return cls(name=response['name'], uid=response['uid'], generation=response['generation'], labels=response.get('labels', {}), annotations=response.get('annotations', {}), createTime=response['createTime'], updateTime=response['updateTime'], deleteTime=response.get('deleteTime'), expireTime=response.get('expireTime'), creator=response.get('creator'), lastModifier=response.get('lastModifier'), client=response.get('client'), clientVersion=response.get('clientVersion'), launchStage=response.get('launchStage', 'GA'), binaryAuthorization=response.get('binaryAuthorization', {}), template=response.get('template'), observedGeneration=response.get('observedGeneration'), terminalCondition=response.get('terminalCondition', {}), conditions=response.get('conditions', []), executionCount=response.get('executionCount', 0), latestCreatedExecution=response['latestCreatedExecution'], reconciling=response.get('reconciling', False), satisfiesPzs=response.get('satisfiesPzs', False), etag=response['etag'])

    @staticmethod
    def create(cr_client: Resource, project: str, location: str, job_id: str, body: Dict) -> Dict:
        """
        Create a job on Cloud Run with the V2 API.

        Args:
            cr_client: The base client needed for interacting with GCP
                Cloud Run V2 API.
            project: The GCP project ID.
            location: The GCP region.
            job_id: The ID of the job to create.
            body: The job body.

        Returns:
            The response from the Cloud Run V2 API.
        """
        request = cr_client.jobs().create(parent=f'projects/{project}/locations/{location}', jobId=job_id, body=body)
        response = request.execute()
        return response

    @staticmethod
    def delete(cr_client: Resource, project: str, location: str, job_name: str) -> Dict:
        """
        Delete a job on Cloud Run with the V2 API.

        Args:
            cr_client (Resource): The base client needed for interacting with GCP
                Cloud Run V2 API.
            project: The GCP project ID.
            location: The GCP region.
            job_name: The name of the job to delete.

        Returns:
            Dict: The response from the Cloud Run V2 API.
        """
        list_executions_request = cr_client.jobs().executions().list(parent=f'projects/{project}/locations/{location}/jobs/{job_name}')
        list_executions_response = list_executions_request.execute()
        for execution_to_delete in list_executions_response.get('executions', []):
            delete_execution_request = cr_client.jobs().executions().delete(name=execution_to_delete['name'])
            delete_execution_request.execute()
            time.sleep(3)
        request = cr_client.jobs().delete(name=f'projects/{project}/locations/{location}/jobs/{job_name}')
        response = request.execute()
        return response

    @staticmethod
    def run(cr_client: Resource, project: str, location: str, job_name: str) -> Dict:
        """
        Run a job on Cloud Run with the V2 API.

        Args:
            cr_client: The base client needed for interacting with GCP
                Cloud Run V2 API.
            project: The GCP project ID.
            location: The GCP region.
            job_name: The name of the job to run.
        """
        request = cr_client.jobs().run(name=f'projects/{project}/locations/{location}/jobs/{job_name}')
        response = request.execute()
        return response

    @staticmethod
    def _is_missing_container(ready_condition: Dict[str, Union[str, Dict]]) -> bool:
        """
        Check if the job is missing a container.

        Args:
            ready_condition: The ready condition for the job.

        Returns:
            Whether the job is missing a container.
        """
        if ready_condition.get('state') == 'CONTAINER_FAILED' and ready_condition.get('reason') == 'ContainerMissing':
            return True
        return False

class ExecutionV2(BaseModel):
    """
    ExecutionV2 is a data model for an execution of a job that will be run on
        Cloud Run API v2.
    """
    name: str
    uid: str
    generation: int
    labels: Dict[str, str]
    annotations: Dict[str, str]
    createTime: str
    startTime: Optional[str]
    completionTime: Optional[str]
    deleteTime: Optional[str]
    expireTime: Optional[str]
    launchStage: str
    job: str
    parallelism: int
    taskCount: int
    template: Dict[str, Union[str, Dict]]
    reconciling: bool
    conditions: List[Dict[str, Union[str, Dict]]]
    observedGeneration: int
    runningCount: int
    succeededCount: int
    failedCount: int
    cancelledCount: int
    retriedCount: int
    logUri: str
    satisfiesPzs: bool
    etag: str

    def is_running(self) -> bool:
        """
        Return whether the execution is running.

        Returns:
            Whether the execution is running.
        """
        return self.completionTime is None

    def succeeded(self) -> bool:
        """Whether or not the Execution completed is a successful state."""
        completed_condition = self.condition_after_completion()
        if completed_condition and completed_condition['state'] == 'CONDITION_SUCCEEDED':
            return True
        return False

    def condition_after_completion(self) -> Optional[Dict[str, Union[str, Dict]]]:
        """
        Return the condition after completion.

        Returns:
            The condition after completion.
        """
        if isinstance(self.conditions, List):
            for condition in self.conditions:
                if condition['type'] == 'Completed':
                    return condition
        return None

    @classmethod
    def get(cls, cr_client: Resource, execution_id: str) -> 'ExecutionV2':
        """
        Get an execution from Cloud Run with the V2 API.

        Args:
            cr_client: The base client needed for interacting with GCP
                Cloud Run V2 API.
            execution_id: The name of the execution to get, in the form of
                projects/{project}/locations/{location}/jobs/{job}/executions
                    /{execution}
        """
        request = cr_client.jobs().executions().get(name=execution_id)
        response = request.execute()
        return cls(name=response['name'], uid=response['uid'], generation=response['generation'], labels=response.get('labels', {}), annotations=response.get('annotations', {}), createTime=response['createTime'], startTime=response.get('startTime'), completionTime=response.get('completionTime'), deleteTime=response.get('deleteTime'), expireTime=response.get('expireTime'), launchStage=response.get('launchStage', 'GA'), job=response['job'], parallelism=response['parallelism'], taskCount=response['taskCount'], template=response['template'], reconciling=response.get('reconciling', False), conditions=response.get('conditions', []), observedGeneration=response.get('observedGeneration'), runningCount=response.get('runningCount'), succeededCount=response.get('succeededCount'), failedCount=response.get('failedCount'), cancelledCount=response.get('cancelledCount'), retriedCount=response.get('retriedCount'), logUri=response['logUri'], satisfiesPzs=response.get('satisfiesPzs', False), etag=response['etag'])
