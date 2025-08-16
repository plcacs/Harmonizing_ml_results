from __future__ import annotations
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from ruamel.yaml import YAML
from prefect.client.schemas.objects import ConcurrencyLimitStrategy
from prefect.client.schemas.schedules import IntervalSchedule
from prefect.utilities._git import get_git_branch, get_git_remote_origin_url
from prefect.utilities.annotations import NotSet
from prefect.utilities.filesystem import create_default_ignore_file
from prefect.utilities.templating import apply_values

def create_default_prefect_yaml(path: Path, name: Optional[str] = None, contents: Optional[Dict] = None) -> bool:
    ...

def configure_project_by_recipe(recipe: str, **formatting_kwargs: Dict) -> Dict:
    ...

def initialize_project(name: Optional[str] = None, recipe: Optional[str] = None, inputs: Optional[Dict] = None) -> List[str]:
    ...

def _format_deployment_for_saving_to_prefect_file(deployment: Dict) -> Dict:
    ...

def _interval_schedule_to_dict(schedule: IntervalSchedule) -> Dict[str, Any]:
    ...

def _save_deployment_to_prefect_file(deployment: Dict, build_steps: Optional[List] = None, push_steps: Optional[List] = None, pull_steps: Optional[List] = None, triggers: Optional[List] = None, sla: Optional[List] = None, prefect_file: Path = Path('prefect.yaml')):
    ...
