import multiprocessing
import os
import random
import shutil
import unittest
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias, Optional, List, Dict, Set, Tuple, Union
from unittest import TestSuite, runner
from unittest.result import TestResult
import orjson
from django.conf import settings
from django.db import ProgrammingError, connections
from django.test import runner as django_runner
from django.test.runner import DiscoverRunner
from django.test.signals import template_rendered
from typing_extensions import override
from scripts.lib.zulip_tools import TEMPLATE_DATABASE_DIR, get_dev_uuid_var_path, get_or_create_dev_uuid_var_path
from zerver.lib import test_helpers
from zerver.lib.partial import partial
from zerver.lib.sqlalchemy_utils import get_sqlalchemy_connection
from zerver.lib.test_fixtures import BACKEND_DATABASE_TEMPLATE
from zerver.lib.test_helpers import append_instrumentation_data, write_instrumentation_reports

random_id_range_start: str = str(random.randint(1, 10000000))

def get_database_id(worker_id: Optional[int] = None) -> str:
    if worker_id:
        return f'{random_id_range_start}_{worker_id}'
    return random_id_range_start

TEST_RUN_DIR: str = get_or_create_dev_uuid_var_path(os.path.join('test-backend', f'run_{get_database_id()}'))
_worker_id: int = 0

class TextTestResult(runner.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failed_tests: List[str] = []

    def addInstrumentation(self, test: Any, data: Dict[str, Any]) -> None:
        append_instrumentation_data(data)

    @override
    def startTest(self, test: unittest.TestCase) -> None:
        TestResult.startTest(self, test)
        self.stream.write(f'Running {test.id()}\n')
        self.stream.flush()

    @override
    def addSuccess(self, *args: Any, **kwargs: Any) -> None:
        TestResult.addSuccess(self, *args, **kwargs)

    @override
    def addError(self, *args: Any, **kwargs: Any) -> None:
        TestResult.addError(self, *args, **kwargs)
        test_name: str = args[0].id()
        self.failed_tests.append(test_name)

    @override
    def addFailure(self, *args: Any, **kwargs: Any) -> None:
        TestResult.addFailure(self, *args, **kwargs)
        test_name: str = args[0].id()
        self.failed_tests.append(test_name)

    @override
    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        TestResult.addSkip(self, test, reason)
        self.stream.write(f'** Skipping {test.id()}: {reason}\n')
        self.stream.flush()

class RemoteTestResult(django_runner.RemoteTestResult):
    def addInstrumentation(self, test: Any, data: Dict[str, Any]) -> None:
        if 'info' in data:
            del data['info']
        self.events.append(('addInstrumentation', self.test_index, data))

def process_instrumented_calls(func: Callable[[Any], None]) -> None:
    for call in test_helpers.INSTRUMENTED_CALLS:
        func(call)

SerializedSubsuite: TypeAlias = Tuple[type[TestSuite], List[str]]
SubsuiteArgs: TypeAlias = Tuple[type['RemoteTestRunner'], int, SerializedSubsuite, bool, bool]

def run_subsuite(args: SubsuiteArgs) -> Tuple[int, List[Any]]:
    test_helpers.INSTRUMENTED_CALLS = []
    _, subsuite_index, subsuite, failfast, buffer = args
    runner = RemoteTestRunner(failfast=failfast, buffer=buffer)
    result = runner.run(subsuite)
    process_instrumented_calls(partial(result.addInstrumentation, None))
    return (subsuite_index, result.events)

def destroy_test_databases(worker_id: Optional[int] = None) -> None:
    for alias in connections:
        connection = connections[alias]
        try:
            if worker_id is not None:
                database_id = get_database_id(worker_id)
                connection.creation.destroy_test_db(suffix=database_id)
            else:
                connection.creation.destroy_test_db()
        except ProgrammingError:
            pass

def create_test_databases(worker_id: int) -> None:
    database_id = get_database_id(worker_id)
    for alias in connections:
        connection = connections[alias]
        connection.creation.clone_test_db(suffix=database_id, keepdb=True)
        settings_dict = connection.creation.get_test_db_clone_settings(database_id)
        connection.settings_dict.update(settings_dict)
        connection.close()

def init_worker(
    counter: multiprocessing.Value,
    initial_settings: Optional[Any] = None,
    serialized_contents: Optional[Any] = None,
    process_setup: Optional[Callable] = None,
    process_setup_args: Optional[Tuple] = None,
    debug_mode: Optional[bool] = None,
    used_aliases: Optional[List[str]] = None
) -> None:
    global _worker_id
    with counter.get_lock():
        counter.value += 1
        _worker_id = counter.value
    from zerver.lib.cache import get_cache_backend
    cache = get_cache_backend(None)
    cache.clear()
    connections.close_all()
    destroy_test_databases(_worker_id)
    create_test_databases(_worker_id)
    initialize_worker_path(_worker_id)

class ParallelTestSuite(django_runner.ParallelTestSuite):
    run_subsuite: Callable[[SubsuiteArgs], Tuple[int, List[Any]]] = run_subsuite
    init_worker: Callable[..., None] = init_worker

def check_import_error(test_name: str) -> None:
    try:
        __import__(test_name)
    except ImportError as exc:
        raise exc from exc

def initialize_worker_path(worker_id: int) -> None:
    worker_path = os.path.join(TEST_RUN_DIR, f'worker_{_worker_id}')
    os.makedirs(worker_path, exist_ok=True)
    settings.TEST_WORKER_DIR = worker_path
    settings.LOCAL_UPLOADS_DIR = get_or_create_dev_uuid_var_path(os.path.join('test-backend', os.path.basename(TEST_RUN_DIR), os.path.basename(worker_path), 'test_uploads'))
    settings.LOCAL_AVATARS_DIR = os.path.join(settings.LOCAL_UPLOADS_DIR, 'avatars')
    settings.LOCAL_FILES_DIR = os.path.join(settings.LOCAL_UPLOADS_DIR, 'files')
    from zerver.lib.upload import upload_backend
    from zerver.lib.upload.local import LocalUploadBackend
    assert isinstance(upload_backend, LocalUploadBackend)

class Runner(DiscoverRunner):
    parallel_test_suite: type[ParallelTestSuite] = ParallelTestSuite

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DiscoverRunner.__init__(self, *args, **kwargs)
        self.templates_rendered: Set[str] = set()
        self.shallow_tested_templates: Set[str] = set()
        template_rendered.connect(self.on_template_rendered)

    @override
    def get_resultclass(self) -> type[TextTestResult]:
        return TextTestResult

    def on_template_rendered(self, sender: Any, context: Dict[str, Any], **kwargs: Any) -> None:
        if hasattr(sender, 'template'):
            template_name = sender.template.name
            if template_name not in self.templates_rendered:
                if context.get('shallow_tested') and template_name not in self.templates_rendered:
                    self.shallow_tested_templates.add(template_name)
                else:
                    self.templates_rendered.add(template_name)
                    self.shallow_tested_templates.discard(template_name)

    def get_shallow_tested_templates(self) -> Set[str]:
        return self.shallow_tested_templates

    @override
    def setup_test_environment(self, *args: Any, **kwargs: Any) -> None:
        settings.DATABASES['default']['NAME'] = BACKEND_DATABASE_TEMPLATE
        filepath = os.path.join(get_dev_uuid_var_path(), TEMPLATE_DATABASE_DIR, get_database_id())
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            if self.parallel > 1:
                for index in range(self.parallel):
                    f.write(get_database_id(index + 1) + '\n')
            else:
                f.write(get_database_id() + '\n')
        if self.parallel == 1:
            initialize_worker_path(0)
        return super().setup_test_environment(*args, **kwargs)

    @override
    def teardown_test_environment(self, *args: Any, **kwargs: Any) -> None:
        if self.parallel > 1:
            for index in range(self.parallel):
                destroy_test_databases(index + 1)
        else:
            destroy_test_databases()
        filepath = os.path.join(get_dev_uuid_var_path(), TEMPLATE_DATABASE_DIR, get_database_id())
        os.remove(filepath)
        try:
            shutil.rmtree(TEST_RUN_DIR)
        except OSError:
            print("Unable to clean up the test run's directory.")
        return super().teardown_test_environment(*args, **kwargs)

    def test_imports(self, test_labels: List[str], suite: TestSuite) -> None:
        prefix = 'unittest.loader._FailedTest.'
        for test_name in get_test_names(suite):
            if test_name.startswith(prefix):
                test_name = test_name.removeprefix(prefix)
                for label in test_labels:
                    if test_name in label:
                        test_name = label
                        break
                check_import_error(test_name)

    @override
    def run_tests(
        self,
        test_labels: List[str],
        failed_tests_path: Optional[str] = None,
        full_suite: bool = False,
        include_webhooks: bool = False,
        **kwargs: Any
    ) -> bool:
        self.setup_test_environment()
        suite = self.build_suite(test_labels)
        self.test_imports(test_labels, suite)
        if self.parallel == 1:
            destroy_test_databases(_worker_id)
            create_test_databases(_worker_id)
        with get_sqlalchemy_connection():
            result = self.run_suite(suite)
            assert isinstance(result, TextTestResult)
        self.teardown_test_environment()
        failed = self.suite_result(suite, result)
        if not failed:
            write_instrumentation_reports(full_suite=full_suite, include_webhooks=include_webhooks)
        if failed_tests_path and result.failed_tests:
            with open(failed_tests_path, 'wb') as f:
                f.write(orjson.dumps(result.failed_tests))
        return failed

def get_test_names(suite: Union[ParallelTestSuite, TestSuite]) -> List[str]:
    if isinstance(suite, ParallelTestSuite):
        return [name for subsuite in suite.subsuites for name in get_test_names(subsuite)]
    else:
        return [t.id() for t in get_tests_from_suite(suite)]

def get_tests_from_suite(suite: TestSuite) -> Iterable[unittest.TestCase]:
    for test in suite:
        if isinstance(test, TestSuite):
            yield from get_tests_from_suite(test)
        else:
            yield test

class RemoteTestRunner(django_runner.RemoteTestRunner):
    resultclass: type[RemoteTestResult] = RemoteTestResult
